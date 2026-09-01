#pragma once

#include <memory>
#include <string>

namespace reader {

class ICursor;
class IResourceLocator;
class ISource;

class ReaderBase {
public:
  ReaderBase(std::shared_ptr<ISource> source,
             std::shared_ptr<IResourceLocator> locator,
             std::string file_path);
  virtual ~ReaderBase() = default;

  virtual int Read() = 0;

protected:
  const std::string& filePath() const noexcept { return file_path_; }
  std::shared_ptr<ISource> source() const noexcept { return source_; }
  std::shared_ptr<IResourceLocator> locator() const noexcept { return locator_; }

private:
  std::shared_ptr<ISource> source_;
  std::shared_ptr<IResourceLocator> locator_;
  std::string file_path_;
};

}  // namespace reader

