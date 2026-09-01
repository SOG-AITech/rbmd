#include <spdlog/spdlog.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "output/include/Logger.hpp"

class TimingStatistics {
private:
    struct Stats {
        double min = std::numeric_limits<double>::max();
        double max = std::numeric_limits<double>::lowest();
        double sum = 0.0;
        double sum_sq = 0.0;  // For variance calculation
        size_t count = 0;
        
        void update(double value) {
            min = std::min(min, value);
            max = std::max(max, value);
            sum += value;
            sum_sq += value * value;
            count++;
        }
        
        double avg() const {
            return count > 0 ? sum / count : 0.0;
        }
        
        double variance() const {
            if (count <= 1) return 0.0;
            double mean = avg();
            return (sum_sq / count) - (mean * mean);
        }
        
        double stddev() const {
            return std::sqrt(variance());
        }
    };
    
    std::unordered_map<std::string, Stats> categories;
    double total_time = 0.0;

    static double max_rank_average(double local_average) {
#ifdef USE_MPI
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Finalized(&finalized);
        }
        if (initialized && !finalized) {
            double global_max = 0.0;
            MPI_Allreduce(&local_average, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);
            return global_max;
        }
#endif
        return local_average;
    }
    
    // Private constructor for singleton
    TimingStatistics() = default;
    
public:
    // Delete copy constructor and assignment operator
    TimingStatistics(const TimingStatistics&) = delete;
    TimingStatistics& operator=(const TimingStatistics&) = delete;
    
    // Get singleton instance
    static TimingStatistics& Instance() {
        static TimingStatistics instance;
        return instance;
    }
    
    // Record time for a category
    void record(const std::string& category, double time_seconds) {
        categories[category].update(time_seconds);
        total_time += time_seconds;
    }

  void print_summary() const {
      // Calculate average total time across all categories
      double avg_total = 0.0;
      for (const auto& category_pair : categories) {
        avg_total += category_pair.second.avg();
      }

      // Build output string
      std::stringstream ss;
      ss << "\n";
      ss << "┌──────────────────┬──────────────────┐\n";
      ss << "│     Category     │ Max Avg Time (s) │\n";
      ss << "├──────────────────┼──────────────────┤\n";

      // Set output format
      ss << std::fixed << std::setprecision(6);

      // Define output order
      const std::vector<std::string> order = {
        "Neighbor-List", "RBL-CacheCheck", "RBL-CoordinateExchange",
        "RBL-CandidateRebuild", "RBL-FilterSample", "Short-Range",
        "RBL-PairKernel", "RBL-LocalReduction", "RBL-MPIAllreduce",
        "RBL-CorrectionApply", "VL-PairKernel", "Long-Range", "Bond", "Angle",
        "Dihedral", "Improper", "SHAKE-A-Cache", "SHAKE-A-InHalo",
        "SHAKE-A-Kernel", "SHAKE-A-Apply", "SHAKE-A-OutHalo",
        "SHAKE-B-Cache", "SHAKE-B-InHalo", "SHAKE-B-Kernel",
        "SHAKE-B-Apply", "SHAKE-B-OutHalo"
    };

      // 按照 order 的顺序输出
      for (const auto& section : order) {
        auto it = categories.find(section);
        const double local_average =
            it != categories.end() ? it->second.avg() : 0.0;
        const double reported_average = max_rank_average(local_average);
        if (reported_average > 0.0) {

          ss << "│ " << std::setw(16) << std::left << section
             << " │ " << std::setw(16) << std::right << reported_average
             << " │\n";
        }
      }

      // 表格底部
      ss << "└──────────────────┴──────────────────┘\n";

      // Log using spdlog
      Logger::Instance().info(ss.str().c_str());
    }

    // Reset statistics
    void reset() {
        categories.clear();
        total_time = 0.0;
    }
};

// Convenience macro for timing code blocks
#define TIME_SECTION(section_name, code_block) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        code_block \
        auto end = std::chrono::high_resolution_clock::now(); \
        std::chrono::duration<double> duration = end - start; \
        TimingStatistics::getInstance().record(section_name, duration.count()); \
    } while(0)
