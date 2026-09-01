#include "thermo_tui_view.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <limits>
#include <sstream>

#include <ftxui/component/event.hpp>
#include <ftxui/dom/canvas.hpp>

using namespace ftxui;  // NOLINT

namespace {

std::string FormatStepLabel(rbmd::Id step) {
  std::ostringstream oss;
  oss << step;
  return oss.str();
}

}  // namespace

struct ThermoStatsTUIView::MetricSeries {
  std::string label;
  std::deque<std::pair<rbmd::Id, double>> history;
  std::deque<double> window;
  bool visible = true;
  Color color = Color::White;
  bool last_anomaly = false;
};

ThermoStatsTUIView::ThermoStatsTUIView(ThermoTUIConfig config)
    : config_(config) {
  palette_ = {
      Color::RGB(255, 99, 132),  Color::RGB(75, 192, 192),
      Color::RGB(54, 162, 235),  Color::RGB(255, 159, 64),
      Color::RGB(153, 102, 255), Color::RGB(201, 203, 207),
      Color::RGB(0, 200, 83),    Color::RGB(255, 214, 0),
      Color::RGB(244, 67, 54),   Color::RGB(0, 188, 212)};

  selector_list_ = Container::Vertical({});
  chart_renderer_ = Renderer([this] { return RenderChartPane(); });
  selector_panel_ =
      Renderer(selector_list_, [this] { return RenderSelectorPane(); });
  layout_ = Container::Horizontal({chart_renderer_, selector_panel_});
}

ThermoStatsTUIView::~ThermoStatsTUIView() { Stop(); }

void ThermoStatsTUIView::Start() {
  if (running_.exchange(true)) {
    return;
  }
  ui_thread_ = std::thread([this] { RunLoop(); });
}

void ThermoStatsTUIView::Stop() {
  if (!running_.exchange(false)) {
    return;
  }

  std::function<void()> exit_loop;
  {
    std::lock_guard lock(screen_mutex_);
    exit_loop = exit_closure_;
  }
  if (exit_loop) {
    exit_loop();
  }

  {
    std::lock_guard lock(screen_mutex_);
    if (screen_ptr_) {
      screen_ptr_->PostEvent(Event::Custom);
    }
  }

  if (ui_thread_.joinable()) {
    ui_thread_.join();
  }
}

void ThermoStatsTUIView::RunLoop() {
  auto screen = ScreenInteractive::Fullscreen();
  {
    std::lock_guard lock(screen_mutex_);
    screen_ptr_ = &screen;
    exit_closure_ = screen.ExitLoopClosure();
  }
  screen.Loop(layout_);
  {
    std::lock_guard lock(screen_mutex_);
    screen_ptr_ = nullptr;
    exit_closure_ = nullptr;
  }
}

void ThermoStatsTUIView::RequestDraw() {
  std::lock_guard lock(screen_mutex_);
  if (screen_ptr_) {
    screen_ptr_->PostEvent(Event::Custom);
  }
}

void ThermoStatsTUIView::OnFrame(const ThermoFrame& frame) {
  {
    std::lock_guard lock(state_mutex_);
    UpdateSeries(frame);
  }
  RequestDraw();
}

void ThermoStatsTUIView::OnReset() {
  std::lock_guard lock(state_mutex_);
  alerts_.clear();
  last_step_ = 0;
  has_critical_error_ = false;
  critical_error_msg_.clear();
  pending_selectors_.clear();
  display_order_.clear();
  for (auto& [_, series] : metrics_) {
    series.history.clear();
    series.window.clear();
    series.last_anomaly = false;
  }
}

void ThermoStatsTUIView::OnShutdown() { Stop(); }

void ThermoStatsTUIView::UpdateSeries(const ThermoFrame& frame) {
  last_step_ = frame.step;
  alerts_.clear();

  for (const auto& key : frame.ordered_keys) {
    auto value_it = frame.values.find(key);
    if (value_it == frame.values.end()) {
      continue;
    }

    const auto label_it = frame.labels.find(key);
    const std::string label =
        label_it != frame.labels.end() ? label_it->second : key;

    auto [it, inserted] = metrics_.try_emplace(key);
    if (inserted) {
      MetricSeries& series = it->second;
      series.label = label;
      const auto color_index = metrics_.size() % palette_.size();
      series.color = palette_[color_index];
      pending_selectors_.push_back(key);
      display_order_.push_back(key);
    }

    MetricSeries& series = it->second;
    series.history.emplace_back(frame.step, value_it->second);
    while (series.history.size() > config_.history_length) {
      series.history.pop_front();
    }

    series.window.push_back(value_it->second);
    if (series.window.size() > config_.anomaly_window) {
      series.window.pop_front();
    }

    int anomaly_type = DetectAnomaly(series);
    series.last_anomaly = (anomaly_type != 0);

    if (series.last_anomaly) {
      std::ostringstream oss;
      oss << series.label;
      if (anomaly_type == 1) {
        oss << " 错误: 数值无效 (NaN/Inf)!";
        if (!has_critical_error_) {
          has_critical_error_ = true;
          critical_error_msg_ = oss.str();
        }
      } else {
        oss << " 异常波动: " << FormatValue(value_it->second);
      }
      alerts_.push_back(oss.str());
    }
  }
}

int ThermoStatsTUIView::DetectAnomaly(const MetricSeries& series) const {
  const double latest = series.window.back();

  if (std::isnan(latest) || std::isinf(latest)) {
    return 1;
  }

  if (series.window.size() < config_.anomaly_window) {
    return 0;
  }

  double sum = 0.0;
  for (double v : series.window) {
    sum += v;
  }
  const double mean = sum / static_cast<double>(series.window.size());

  double variance = 0.0;
  for (double v : series.window) {
    const double delta = v - mean;
    variance += delta * delta;
  }
  variance /= static_cast<double>(series.window.size());
  const double stddev = std::sqrt(std::max(variance, 0.0));
  if (stddev < 1e-12) {
    return 0;
  }

  if (std::fabs(latest - mean) > config_.anomaly_sigma * stddev) {
    return 2;
  }
  return 0;
}

void ThermoStatsTUIView::ApplyPendingSelectors() {
  for (const auto& key : pending_selectors_) {
    auto it = metrics_.find(key);
    if (it == metrics_.end()) {
      continue;
    }
    selector_list_->Add(Checkbox(it->second.label, &it->second.visible));
  }
  pending_selectors_.clear();
}

Element ThermoStatsTUIView::RenderChartPane() {
  Element alerts;
  Element graph;
  std::string step_label;
  bool is_critical = false;
  std::string critical_msg;

  {
    std::lock_guard lock(state_mutex_);
    ApplyPendingSelectors();
    alerts = BuildAlertsLocked();
    graph = BuildGraphLocked();
    step_label = FormatStepLabel(last_step_);
    is_critical = has_critical_error_;
    critical_msg = critical_error_msg_;
  }

  auto main_content = vbox({
      text("热力学监控 (Step " + step_label + ")") | bold |
          color(is_critical ? Color::Red : Color::Yellow),
      separator(),
      alerts,
      separator(),
      graph | flex,
  });

  if (is_critical) {
    main_content = vbox({
        text(" ⚠️  CRITICAL SIMULATION ERROR ⚠️ ") | bold | color(Color::Red) |
            blink | center,
        text(critical_msg) | color(Color::Red) | center,
        separator() | color(Color::Red),
        main_content | flex,
    });
    return main_content | flex | borderStyled(Color::Red);
  }

  return main_content | flex | border;
}

Element ThermoStatsTUIView::RenderSelectorPane() {
  {
    std::lock_guard lock(state_mutex_);
    ApplyPendingSelectors();
  }
  auto selectors = selector_list_->Render() | vscroll_indicator | frame;

  Element current_values;
  {
    std::lock_guard lock(state_mutex_);
    current_values = BuildLatestValuesLocked();
  }

  return vbox({
             text("指标选择") | bold,
             separator(),
             selectors,
             separator(),
             text("当前值") | bold,
             current_values,
         }) |
         size(WIDTH, EQUAL, 32) | border;
}

Element ThermoStatsTUIView::BuildAlertsLocked() {
  if (alerts_.empty()) {
    return text("状态: 正常") | color(Color::Green);
  }

  Elements rows;
  for (const auto& msg : alerts_) {
    rows.push_back(text("告警: " + msg) | color(Color::Red));
  }
  return vbox(std::move(rows));
}

Element ThermoStatsTUIView::BuildLegendLocked() {
  return vbox({
             text("提示"),
             text("· 空格/Enter 在列表中切换指标可见性"),
             text("· ESC 退出 TUI"),
         }) |
         color(Color::GrayLight);
}

Element ThermoStatsTUIView::BuildLatestValuesLocked() {
  Elements rows;
  for (const auto& [_, series] : metrics_) {
    if (series.history.empty()) {
      continue;
    }
    std::ostringstream oss;
    oss << std::setw(10) << series.label << " : "
        << FormatValue(series.history.back().second);
    auto row = text(oss.str()) | color(series.color);
    if (!series.visible) {
      row = row | dim;
    }
    if (series.last_anomaly) {
      row = row | bgcolor(Color::Red) | color(Color::White);
    }
    rows.push_back(row);
  }
  if (rows.empty()) {
    rows.push_back(text("暂无数据") | dim);
  }
  return vbox(std::move(rows)) | frame;
}

Element ThermoStatsTUIView::BuildGraphLocked() {
  Elements rows;
  const int height = 12;

  bool has_data = false;

  for (const auto& key : display_order_) {
    auto it = metrics_.find(key);
    if (it == metrics_.end()) continue;
    const auto& series = it->second;

    if (!series.visible || series.history.empty()) {
      continue;
    }
    has_data = true;

    double min_value = std::numeric_limits<double>::max();
    double max_value = std::numeric_limits<double>::lowest();
    for (const auto& [_, value] : series.history) {
      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
    }

    if (std::fabs(max_value - min_value) < 1e-12) {
      const double delta = std::max(1.0, std::fabs(max_value) * 0.1);
      min_value -= delta;
      max_value += delta;
    }

    auto project_y = [&](double value) {
      const double normalized =
          std::clamp((value - min_value) / (max_value - min_value), 0.0, 1.0);
      const int y = static_cast<int>(normalized * (height - 1));
      return height - 1 - y;
    };

    const int series_size = static_cast<int>(series.history.size());
    const int width =
        std::max(2, std::min(static_cast<int>(config_.history_length), series_size));
    const int start = std::max(0, series_size - width);
    Canvas c(width, height);

    if (series_size > 1) {
      for (int i = start + 1; i < series_size; ++i) {
        const auto [step0, value0] = series.history[static_cast<std::size_t>(i - 1)];
        const auto [step1, value1] = series.history[static_cast<std::size_t>(i)];
        (void)step0;
        (void)step1;
        const int x0 = i - start - 1;
        const int x1 = i - start;
        const int y0 = project_y(value0);
        const int y1 = project_y(value1);
        c.DrawPointLine(x0, y0, x1, y1, series.color);
      }
    }

    const auto [latest_step, latest_value] = series.history.back();
    (void)latest_step;
    c.DrawPointLine(width - 1, project_y(latest_value), width - 1,
                    project_y(latest_value), series.color);

    auto label_elem = text(series.label) | size(WIDTH, EQUAL, 10) | bold;

    auto value_elem = text(FormatValue(latest_value)) | align_right |
                      size(WIDTH, EQUAL, 15) | color(series.color);

    auto graph_elem = canvas(std::move(c));

    auto row = hbox({label_elem, separator(), value_elem, separator(),
                     graph_elem});

    if (series.last_anomaly) {
      row = row | bgcolor(Color::Red);
    }

    rows.push_back(row);
    rows.push_back(separatorLight());
  }

  if (!has_data) {
    return text("等待数据...") | dim | center;
  }

  return vbox(std::move(rows)) | vscroll_indicator | yframe | flex;
}

std::string ThermoStatsTUIView::FormatValue(double value) const {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(3) << value;
  return oss.str();
}
