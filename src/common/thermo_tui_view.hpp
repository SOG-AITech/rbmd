#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include "thermo_view.h"

struct ThermoTUIConfig {
  size_t history_length = 200;
  size_t anomaly_window = 32;
  double anomaly_sigma = 10.0;
  int chart_height = 28;
};

class ThermoStatsTUIView : public IThermoView,
                           public std::enable_shared_from_this<ThermoStatsTUIView> {
 public:
  explicit ThermoStatsTUIView(ThermoTUIConfig config = {});
  ~ThermoStatsTUIView() override;

  void Start();
  void Stop();

  void OnFrame(const ThermoFrame& frame) override;
  void OnReset() override;
  void OnShutdown() override;

 private:
  struct MetricSeries;
  struct Snapshot;

  void RunLoop();
  void RequestDraw();

  void UpdateSeries(const ThermoFrame& frame);
  int DetectAnomaly(const MetricSeries& series) const;
  void ApplyPendingSelectors();

  ftxui::Element RenderChartPane();
  ftxui::Element RenderSelectorPane();

  ftxui::Element BuildAlertsLocked();
  ftxui::Element BuildGraphLocked();
  ftxui::Element BuildLegendLocked();
  ftxui::Element BuildLatestValuesLocked();

  std::string FormatValue(double value) const;

  ThermoTUIConfig config_;
  std::vector<ftxui::Color> palette_;

  std::mutex state_mutex_;
  std::map<std::string, MetricSeries> metrics_;
  std::vector<std::string> display_order_;
  std::vector<std::string> pending_selectors_;
  std::vector<std::string> alerts_;
  bool has_critical_error_ = false;
  std::string critical_error_msg_;
  rbmd::Id last_step_ = 0;

  ftxui::Component selector_list_;
  ftxui::Component selector_panel_;
  ftxui::Component chart_renderer_;
  ftxui::Component layout_;

  std::thread ui_thread_;
  std::mutex screen_mutex_;
  ftxui::ScreenInteractive* screen_ptr_ = nullptr;
  std::function<void()> exit_closure_;
  std::atomic<bool> running_{false};
};
