#pragma once

// A narrow output adapter for a previously opened, caller-verified REGULAR file.
// Does not own/close the ofstream. Create, write, Finish and destroy on ONE thread.
// All readers must wait for successful Finish + caller close + MD exit success.
// ostream::flush feeds ordinary deflate and flushes the underlying filebuf; it
// deliberately does NOT issue Z_SYNC_FLUSH, so an active file is not a readable
// checkpoint. Do not seek/tell or move/swap/reopen/close the ofstream while attached.
// Finish() throws on any stream/compressor/sink failure and ALWAYS detaches first.
// A forgotten explicit Finish terminates: destructor cannot hide an output error.

#include <zlib.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <thread>

namespace regular_gzip_detail {

class Buffer final : public std::streambuf {
 public:
  static constexpr std::size_t kBufferBytes = 64u * 1024u;
  enum class Error { None, WrongThread, Zlib, ShortWrite, SinkException,
                     SinkSync, WriteAfterFinish, InvalidWrite, OutputLimit };

  explicit Buffer(std::streambuf* sink, int level,
                  std::uint64_t output_limit = std::numeric_limits<std::uint64_t>::max())
      : sink_(sink), owner_(std::this_thread::get_id()), output_limit_(output_limit) {
    if (!sink_) throw std::invalid_argument("gzip: null output sink");
    if (level < 0 || level > 9)
      throw std::invalid_argument("gzip: level must be in [0,9]");
    const int rc = deflateInit2(&z_, level, Z_DEFLATED, 15 + 16, 8,
                                Z_DEFAULT_STRATEGY);
    if (rc != Z_OK) throw std::runtime_error("gzip: deflateInit2 failed (" +
                                              std::to_string(rc) + ")");
    live_ = true;
    setp(input_.data(), input_.data() + input_.size());
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() noexcept override { if (live_) (void)deflateEnd(&z_); }

  bool Finish() noexcept {
    if (finish_attempted_) return finish_ok_;
    finish_attempted_ = true;
    bool ok = CheckOwner() && error_ == Error::None;
    if (ok) ok = DeflatePending(Z_FINISH) && stream_ended_;
    if (ok) ok = SyncSink();
    if (live_) {
      const int rc = deflateEnd(&z_);
      live_ = false;
      if (rc != Z_OK) { SetError(Error::Zlib, rc); ok = false; }
    }
    // Disallow cheap single-character writes through the put area after finish.
    setp(nullptr, nullptr);
    finish_ok_ = ok && error_ == Error::None;
    return finish_ok_;
  }

  bool Good() const noexcept { return error_ == Error::None; }
  bool FinishedSuccessfully() const noexcept { return finish_ok_; }
  std::uint64_t InputBytes() const noexcept {
    const auto pending = pbase() && pptr()
      ? static_cast<std::uint64_t>(pptr() - pbase()) : 0;
    return input_bytes_ + pending;
  }
  std::uint64_t CompressedBytes() const noexcept { return output_bytes_; }
  Error GetError() const noexcept { return error_; }
  int ZlibStatus() const noexcept { return zlib_status_; }
  std::exception_ptr SinkException() const noexcept { return sink_exception_; }
  std::string ErrorMessage() const {
    switch (error_) {
      case Error::None: return "gzip: no compressor error";
      case Error::WrongThread: return "gzip: output used by a non-owner thread";
      case Error::Zlib: return "gzip: deflate failure (" +
                                  std::to_string(zlib_status_) + ")";
      case Error::ShortWrite: return "gzip: short write to regular-file sink";
      case Error::SinkException: return "gzip: regular-file sink threw";
      case Error::SinkSync: return "gzip: regular-file sink flush failed";
      case Error::WriteAfterFinish: return "gzip: write/flush after Finish";
      case Error::InvalidWrite: return "gzip: invalid input write";
      case Error::OutputLimit: return "gzip: registered compressed-output byte limit exceeded";
    }
    return "gzip: unknown failure";
  }

 protected:
  std::streamsize xsputn(const char* data, std::streamsize size) override {
    if (!CheckOwner()) return 0;
    if (size < 0 || (size > 0 && !data)) {
      SetError(Error::InvalidWrite); return 0;
    }
    if (finish_attempted_) {
      if (size == 0) return 0;
      SetError(Error::WriteAfterFinish); return 0;
    }
    if (error_ != Error::None) return 0;
    const std::streamsize requested = size;
    while (size > 0) {
      if (pptr() == epptr() && !DeflatePending(Z_NO_FLUSH))
        return requested - size;
      const std::size_t room = static_cast<std::size_t>(epptr() - pptr());
      const auto chunk = static_cast<std::size_t>(std::min<std::streamsize>(
          size, static_cast<std::streamsize>(room)));
      std::memcpy(pptr(), data, chunk);
      pbump(static_cast<int>(chunk)); // chunk <=64KiB, safely below INT_MAX.
      data += chunk;
      size -= static_cast<std::streamsize>(chunk);
    }
    return requested;
  }

  int_type overflow(int_type ch) override {
    if (!CheckOwner()) return traits_type::eof();
    if (finish_attempted_) {
      SetError(Error::WriteAfterFinish); return traits_type::eof();
    }
    if (error_ != Error::None || !DeflatePending(Z_NO_FLUSH))
      return traits_type::eof();
    if (traits_type::eq_int_type(ch, traits_type::eof()))
      return traits_type::not_eof(ch);
    *pptr() = traits_type::to_char_type(ch);
    pbump(1);
    return ch;
  }

  int sync() override {
    if (!CheckOwner()) return -1;
    if (finish_attempted_) {
      SetError(Error::WriteAfterFinish); return -1;
    }
    return error_ == Error::None && DeflatePending(Z_NO_FLUSH) && SyncSink()
               ? 0 : -1;
  }

 private:
  void SetError(Error e, int zlib_status = Z_OK) noexcept {
    if (error_ == Error::None) { error_ = e; zlib_status_ = zlib_status; }
  }
  bool CheckOwner() noexcept {
    if (std::this_thread::get_id() == owner_) return true;
    SetError(Error::WrongThread); return false;
  }
  bool WriteCompressed(std::size_t size) noexcept {
    if (size == 0) return true;
    if (output_bytes_ > output_limit_ || size > output_limit_ - output_bytes_) {
      SetError(Error::OutputLimit); return false;
    }
    try {
      const auto want = static_cast<std::streamsize>(size);
      const auto wrote = sink_->sputn(output_.data(), want);
      if (wrote > 0)
        output_bytes_ += static_cast<std::uint64_t>(wrote);
      if (wrote != want) { SetError(Error::ShortWrite); return false; }
      return true;
    } catch (...) {
      sink_exception_ = std::current_exception();
      SetError(Error::SinkException); return false;
    }
  }
  bool SyncSink() noexcept {
    try {
      if (sink_->pubsync() == 0) return true;
      SetError(Error::SinkSync); return false;
    } catch (...) {
      sink_exception_ = std::current_exception();
      SetError(Error::SinkException); return false;
    }
  }
  bool DeflatePending(int flush) noexcept {
    if (!live_ || error_ != Error::None) return false;
    const std::size_t pending = static_cast<std::size_t>(pptr() - pbase());
    if (pending == 0 && flush == Z_NO_FLUSH) return true;
    z_.next_in = reinterpret_cast<Bytef*>(pbase());
    z_.avail_in = static_cast<uInt>(pending); // input buffer <=64KiB.
    input_bytes_ += static_cast<std::uint64_t>(pending);
    // Count direct sputc/character writes too. No concurrent owner may write
    // while deflate reads this same input array.
    setp(input_.data(), input_.data() + input_.size());
    while (true) {
      z_.next_out = reinterpret_cast<Bytef*>(output_.data());
      z_.avail_out = static_cast<uInt>(output_.size());
      const auto before_in = z_.avail_in;
      const int rc = deflate(&z_, flush);
      const std::size_t produced = output_.size() - z_.avail_out;
      // Z_BUF_ERROR with no progress after an exactly-full output block is a
      // benign drain completion for ordinary deflate, never for Z_FINISH.
      if (rc == Z_BUF_ERROR && flush == Z_NO_FLUSH && z_.avail_in == 0 &&
          produced == 0 && before_in == 0) break;
      if (rc != Z_OK && rc != Z_STREAM_END) {
        SetError(Error::Zlib, rc); return false;
      }
      if (!WriteCompressed(produced)) return false;
      if (rc == Z_STREAM_END) {
        stream_ended_ = true;
        if (flush != Z_FINISH || z_.avail_in != 0) {
          SetError(Error::Zlib, Z_STREAM_ERROR); return false;
        }
        break;
      }
      if (flush == Z_FINISH) {
        if (before_in == z_.avail_in && produced == 0) {
          SetError(Error::Zlib, Z_BUF_ERROR); return false;
        }
        continue; // Require Z_STREAM_END, including the gzip CRC/footer.
      }
      if (z_.avail_in == 0 && z_.avail_out != 0) break;
    }
    setp(input_.data(), input_.data() + input_.size());
    return true;
  }

  std::streambuf* sink_;
  const std::thread::id owner_;
  const std::uint64_t output_limit_;
  z_stream z_{};
  std::array<char, kBufferBytes> input_{};
  std::array<char, kBufferBytes> output_{};
  bool live_ = false, finish_attempted_ = false, stream_ended_ = false;
  bool finish_ok_ = false;
  Error error_ = Error::None;
  int zlib_status_ = Z_OK;
  std::uint64_t input_bytes_ = 0, output_bytes_ = 0;
  std::exception_ptr sink_exception_;
};

} // namespace regular_gzip_detail

class RegularGzipStream final {
 public:
  explicit RegularGzipStream(
      std::ofstream& file, int level = 1,
      std::uint64_t output_limit = std::numeric_limits<std::uint64_t>::max())
      : file_(file), original_(Validate(file)), buffer_(original_, level, output_limit) {
    Attach();
  }
#ifdef REGULAR_GZIP_STREAM_TESTING
  // Fault injection only. Production always uses the original regular filebuf.
  RegularGzipStream(std::ofstream& file, std::streambuf& injected_sink,
                    int level = 1)
      : file_(file), original_(Validate(file)), buffer_(&injected_sink, level) {
    Attach();
  }
#endif
  RegularGzipStream(const RegularGzipStream&) = delete;
  RegularGzipStream& operator=(const RegularGzipStream&) = delete;
  RegularGzipStream(RegularGzipStream&&) = delete;
  RegularGzipStream& operator=(RegularGzipStream&&) = delete;

  ~RegularGzipStream() noexcept {
    if (!finish_attempted_) {
      // A destructor cannot report a finalization error. A missing Finish must
      // fail the process rather than accidentally produce MD COMPLETED0.
      DetachNoThrow(std::ios::badbit);
      std::terminate();
    }
  }

  void Finish() {
    if (finish_attempted_) {
      if (!finish_ok_) throw std::runtime_error(failure_message_);
      return;
    }
    finish_attempted_ = true;
    const auto old_state = file_.rdstate();
    const bool attachment_ok = file_.is_open() &&
              static_cast<std::ostream&>(file_).rdbuf() == &buffer_;
    const bool buffer_ok = attachment_ok && buffer_.Finish();
    const bool stream_ok =
      (old_state & (std::ios::badbit | std::ios::failbit)) == 0;
    const bool ok = attachment_ok && buffer_ok && stream_ok;
    // Restore BEFORE allocating a diagnostic string (even bad_alloc must not
    // leave the ofstream pointing at a soon-to-be-destroyed gzip buffer).
    DetachNoThrow(old_state | (ok ? std::ios::goodbit : std::ios::badbit));
    finish_ok_ = ok;
    if (!ok) {
      if (!attachment_ok)
        failure_message_ = "gzip: ofstream closed or its rdbuf replaced before Finish";
      else if (!buffer_.Good()) failure_message_ = buffer_.ErrorMessage();
      else if (!stream_ok)
        failure_message_ = "gzip: ofstream already had a failed output operation";
      else failure_message_ = "gzip: finalization failed";
      throw std::runtime_error(failure_message_);
    }
  }

  bool FinishedSuccessfully() const noexcept { return finish_ok_; }
  std::uint64_t InputBytes() const noexcept { return buffer_.InputBytes(); }
  std::uint64_t CompressedBytes() const noexcept { return buffer_.CompressedBytes(); }

 private:
  static std::streambuf* Validate(std::ofstream& file) {
    if (!file.is_open() || !file.good())
      throw std::invalid_argument("gzip: requires an open, good regular-file ofstream");
    auto* original = static_cast<std::ostream&>(file).rdbuf();
    if (original != file.rdbuf())
      throw std::invalid_argument("gzip: ofstream already has a replacement rdbuf");
    return original;
  }
  void Attach() {
    static_cast<std::ostream&>(file_).rdbuf(&buffer_);
    attached_ = true;
  }
  void DetachNoThrow(std::ios::iostate state) noexcept {
    if (!attached_) return;
    const auto mask = file_.exceptions();
    try {
      file_.exceptions(std::ios::goodbit);
      static_cast<std::ostream&>(file_).rdbuf(original_);
      attached_ = false;
      file_.clear(state);
      try { file_.exceptions(mask); } catch (...) {
        // exceptions(mask) leaves the mask installed. Finish will report the
        // captured failure after the stream has been safely detached.
      }
    } catch (...) {
      // Restoring an attached stream is a non-negotiable lifetime invariant.
      std::terminate();
    }
  }

  std::ofstream& file_;
  std::streambuf* original_;
  regular_gzip_detail::Buffer buffer_;
  bool attached_ = false, finish_attempted_ = false, finish_ok_ = false;
  std::string failure_message_;
};
