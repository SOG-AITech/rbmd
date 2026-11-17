#pragma once
#include <spdlog/spdlog.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
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
#include "common/mpi_root_guard.hpp"

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
            return std::sqrt(std::max(0.0, variance()));
        }
    };
    
    std::unordered_map<std::string, Stats> categories;

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
    }

  void print_summary(const std::string& context = "") const {
      std::ostringstream ss;
      std::ostringstream table;
      table << "\nCategory                         | Max Avg Time (s)\n";
      table << std::fixed << std::setprecision(9);
      ss << "# RBMD timing; seconds; " << context << '\n'
         << "# Step-Wall: device synchronized before/after integration; excludes external output.\n"
         << "# Loop wall includes output/diagnostic overhead; frozen evaluations are not MD steps.\n"
         << "# *KernelGPU / Long-ComputeGPU: isolated event intervals, exclude MPI and transfers.\n"
         << "# *Workflow-Wall: synchronized inclusive work, including internal MPI/transfers.\n"
         << "# *OverlapSpan and legacy categories are diagnostic spans, not additive stage costs.\n"
         << "# Rank means are aggregated separately; never sum maxima of different stages.\n"
         << "category,count_min,count_max,rank_mean_min_s,rank_mean_avg_s,rank_mean_max_s\n";
      ss << std::setprecision(17);
      const std::vector<std::string> order = {
        "Step-Wall", "Frozen-Evaluation-Wall", "Force-Evaluation-Wall", "Pair-Workflow-Wall", "Kspace-Workflow-Wall",
        "Pair-KernelGPU", "RBL-CorrectionGPU", "Force-AssembleGPU",
        "Long-ComputeGPU", "RBE-VirialEnergyGPU",
        "Pair-OverlapSpan", "Kspace-OverlapSpan", "Bonded-OverlapSpan",
        "Neighbor-List", "RBL-CacheCheck", "RBL-DirectSample", "RBL-RebuildExchange", "RBL-CoordinateExchange",
        "RBL-CandidateRebuild", "RBL-FilterSample", "Short-Range",
        "RBL-PairKernel", "RBL-LocalReduction", "RBL-MPIAllreduce",
        "RBL-CorrectionApply", "VL-PairKernel", "Long-Range", "Bond", "Angle",
        "Dihedral", "Improper", "Bonded", "Force-Step",
        "RBE-SampleHost", "RBE-MPILaunch", "RBE-SampleH2D", "RBE-RhoD2H", "RBE-RhoH2D", "RBE-EnergyD2H", "RBE-RhoGPU", "RBE-MPIWait", "RBE-ForceGPU",
        "RBL-CorrectionMPIWait", "Force-EnergyMPIWait", "SHAKE-A-Cache", "SHAKE-A-InHalo",
        "SHAKE-A-Kernel", "SHAKE-A-Apply", "SHAKE-A-OutHalo",
        "SHAKE-B-Cache", "SHAKE-B-InHalo", "SHAKE-B-Kernel",
        "SHAKE-B-Apply", "SHAKE-B-OutHalo"
    };

      for (const auto& section : order) {
        const auto it = categories.find(section);
        const bool present = it != categories.end() && it->second.count;
        const double count = present ? static_cast<double>(it->second.count) : 0;
        const double mean = present ? it->second.avg() : 0;
        double lo[2] = {count, present ? mean : std::numeric_limits<double>::infinity()};
        double hi[2] = {count, mean};
        double sums[2] = {present ? 1.0 : 0.0, mean};
#ifdef USE_MPI
        int initialized = 0, finalized = 0;
        MPI_Initialized(&initialized);
        if (initialized) MPI_Finalized(&finalized);
        if (initialized && !finalized) {
          MPI_Allreduce(MPI_IN_PLACE, lo, 2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
          MPI_Allreduce(MPI_IN_PLACE, hi, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
          MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
#endif
        if (sums[0] > 0) {
          ss << section << ',' << lo[0] << ',' << hi[0] << ',' << lo[1] << ','
             << sums[1] / sums[0] << ',' << hi[1] << '\n';
          table << "│ " << std::setw(32) << std::left << section
                << " │ " << std::setw(16) << std::right << hi[1] << " │\n";
        }
      }
      Logger::Instance().info("{}", table.str());
      if (rbmd::mpi::ShouldWriteRootOnlyOutput()) {
        std::ofstream out("time.txt", std::ios::trunc);
        if (!out) throw std::runtime_error("Cannot create time.txt");
        out << ss.str();
        if (!out) throw std::runtime_error("Failed writing time.txt");
      }
    }

    void write_csv(const std::string& path) const {
        std::ofstream out(path, std::ios::trunc);
        if (!out) throw std::runtime_error("Cannot write timing CSV: " + path);
        out << "category,count,sum_seconds,mean_seconds,min_seconds,max_seconds\n";
        out << std::setprecision(17);
        std::vector<std::string> names;
        for (const auto& item : categories) names.push_back(item.first);
        std::sort(names.begin(), names.end());
        for (const auto& name : names) {
            const auto& s = categories.at(name);
            out << name << ',' << s.count << ',' << s.sum << ',' << s.avg()
                << ',' << s.min << ',' << s.max << '\n';
        }
        if (!out) throw std::runtime_error("Failed writing timing CSV: " + path);
    }

    // Reset statistics
    void reset() {
        categories.clear();
    }
};

// Convenience macro for timing code blocks
#define TIME_SECTION(section_name, code_block) \
    do { \
        auto start = std::chrono::high_resolution_clock::now(); \
        code_block \
        auto end = std::chrono::high_resolution_clock::now(); \
        std::chrono::duration<double> duration = end - start; \
        TimingStatistics::Instance().record(section_name, duration.count()); \
    } while(0)
