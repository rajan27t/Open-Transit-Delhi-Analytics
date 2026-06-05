import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget

# ==========================================
# 0. AESTHETICS & STYLING (CSS & MATPLOTLIB)
# ==========================================
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;600;700&display=swap');
    
    body {
        font-family: 'Inter', sans-serif !important;
        background-color: #f4f7f6;
        color: #2c3e50;
    }
    .card {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        border: none;
        margin-bottom: 15px;
    }
    .nav-pills .nav-link.active, .nav-pills .show>.nav-link {
        background-color: #1f77b4;
        border-radius: 8px;
    }
    h2, h4, h5 {
        font-weight: 600;
        color: #1a252f;
    }
    .insight-box {
        background-color: #eef4f8;
        border-left: 4px solid #1f77b4;
        padding: 12px;
        margin-top: 10px;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
        font-style: italic;
    }
</style>
"""

banner_html = """
<div style="width: 100%; height: 130px; background: linear-gradient(135deg, #1f77b4 0%, #0c2d48 100%); 
            border-radius: 12px; margin-bottom: 25px; display: flex; align-items: center; justify-content: center;
            color: white; font-family: 'Inter', sans-serif; box-shadow: 0 6px 15px rgba(31, 119, 180, 0.3);">
    <div style="text-align: center;">
        <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700; letter-spacing: 0.5px;">Delhi Transit Analytics</h1>
        <p style="margin: 5px 0 0 0; font-size: 1.1rem; font-weight: 300; opacity: 0.9;">Performance & Reliability Dashboard</p>
    </div>
</div>
"""

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Arial", "sans-serif"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.color": "#b0bec5",
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "text.color": "#2c3e50",
})

COLORS = {
    "blue": "#1f77b4", "orange": "#ff7f0e", "green": "#2ca02c",
    "red": "#d62728", "purple": "#9467bd"
}

SCHEDULED_ROUTES = ['148', '1174', '1391', '720', '37', '445', '486', '537', '627', '585', '681', '9', '126', '850']

# ==========================================
# 1. GLOBAL SETUP & DATA LOADING
# ==========================================
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
slot_order = ["Morning Off Peak", "Morning Peak", "Off Peak", "Evening Peak", "Evening Off Peak"]

def time_slot(hour):
    if 0 <= hour < 8: return "Morning Off Peak"
    elif 8 <= hour < 11: return "Morning Peak"
    elif 11 <= hour < 17: return "Off Peak"
    elif 17 <= hour < 20: return "Evening Peak"
    else: return "Evening Off Peak"

def load_and_process_data():
    try:
        df_raw = pd.read_csv("ALL_ROUTES_master_summary.csv")
    except FileNotFoundError:
        df_raw = pd.DataFrame(columns=["route_id", "start_time", "date", "travel_time_min"])
        
    if not df_raw.empty:
        df_raw["route_id"] = df_raw["route_id"].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        df_raw["start_time_dt"] = pd.to_datetime(df_raw["start_time"], format='mixed', dayfirst=True)
        df_raw["date"] = pd.to_datetime(df_raw["date"], format='mixed', dayfirst=True)
        df_raw["hour"] = df_raw["start_time_dt"].dt.hour
        df_raw["time_period"] = df_raw["hour"].apply(time_slot)
        df_raw["day"] = pd.Categorical(df_raw["start_time_dt"].dt.day_name(), categories=day_order, ordered=True)
        df_raw["real_min"] = df_raw["start_time_dt"].dt.hour * 60 + df_raw["start_time_dt"].dt.minute

    route_days_count = df_raw.dropna(subset=['day']).groupby('route_id')['day'].nunique() if not df_raw.empty else pd.Series()
    reliability_routes = route_days_count[route_days_count == 7].index.astype(str).tolist()

    try:
        sched_df = pd.read_csv("MASTER_SCHEDULE.csv")
        sched_df["route_id"] = sched_df["route_id"].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
        valid_routes = set(df_raw['route_id']).intersection(set(sched_df['route_id']))
        df_otp = df_raw[df_raw['route_id'].isin(valid_routes)].copy()
        sched_df = sched_df[sched_df['route_id'].isin(valid_routes)].copy()

        sched_df["start_time_dt"] = pd.to_datetime(sched_df["start_time"], format="%H:%M:%S")
        sched_df["sched_min"] = sched_df["start_time_dt"].dt.hour * 60 + sched_df["start_time_dt"].dt.minute

        df_otp = df_otp.sort_values('real_min')
        sched_df = sched_df.sort_values('sched_min')

        merged_df = pd.merge_asof(
            df_otp,
            sched_df[['route_id', 'sched_min']],
            left_on='real_min',
            right_on='sched_min',
            by='route_id',
            direction='nearest'
        )
        df_otp = merged_df.copy()
        df_otp["sched_deviation"] = df_otp["real_min"] - df_otp["sched_min"]
        
        df_otp = df_otp.sort_values(["route_id", "date", "real_min"])
        df_otp["real_headway"] = df_otp.groupby(["route_id", "date"])["real_min"].diff()
        df_otp["time_slot"] = pd.Categorical(df_otp["time_period"], categories=slot_order, ordered=True)
    except FileNotFoundError:
        df_otp = pd.DataFrame()

    return df_raw, df_otp, reliability_routes

try:
    master_raw_df, master_otp_df, RELIABILITY_ROUTES = load_and_process_data()
    otp_choices = ["All Routes"] + sorted(SCHEDULED_ROUTES, key=lambda x: int(x) if x.isdigit() else x)
    rel_choices = ["All Routes"] + sorted(RELIABILITY_ROUTES, key=lambda x: int(x) if x.isdigit() else x)
except Exception as e:
    master_raw_df = pd.DataFrame()
    master_otp_df = pd.DataFrame()
    otp_choices = ["All Routes"]
    rel_choices = ["All Routes"]


# ==========================================
# 2. USER INTERFACE (UI)
# ==========================================
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Dashboard Filters"),
        ui.panel_conditional(
            "input.main_tabs === 'tab_otp'",
            ui.input_select("route_otp", "Select Route ID", choices=otp_choices)
        ),
        ui.panel_conditional(
            "input.main_tabs === 'tab_rel'",
            ui.input_select("route_rel", "Select Route ID", choices=rel_choices)
        ),
        title="Settings"
    ),
    ui.tags.head(ui.HTML(custom_css)),
    ui.HTML(banner_html),
    ui.navset_card_pill(
        ui.nav_panel("On-Time Performance (OTP)",
            ui.layout_columns(
                ui.value_box("Overall OTP Score", ui.output_text("kpi_otp"), theme="primary"),
                ui.value_box("Overall Average Schedule Deviation at origin stop", ui.output_text("kpi_dev"), theme="info"),
                ui.value_box("Avg Real Headway", ui.output_text("kpi_headway"), theme="warning"),
                ui.value_box("Estimated Average Wait Time", ui.output_text("kpi_wait"), theme="danger"),
            ),
            ui.hr(),
            ui.navset_card_tab(
                ui.nav_panel("OTP Analysis", 
                    ui.layout_columns(ui.output_plot("plot_otp_day"), ui.output_plot("plot_otp_slot")),
                    ui.div(ui.output_text("text_otp_analysis"), class_="insight-box")
                ),
                ui.nav_panel("Headway Adherence", 
                    ui.output_plot("plot_cov_heatmap")),
                ui.nav_panel("Waiting Time Analysis", 
                    ui.layout_columns(ui.output_plot("plot_wait_day"), ui.output_plot("plot_wait_slot")),
                    ui.div(ui.output_text("text_wait_analysis"), class_="insight-box")
                )
            ),
            value="tab_otp"
        ),
        ui.nav_panel("Route Reliability",
            ui.navset_card_tab(
                ui.nav_panel("Interactive Trends", 
                    output_widget("interactive_scatter"), 
                    output_widget("interactive_boxplot"),
                    ui.div(ui.output_text("text_interactive_trends"), class_="insight-box")
                ),
                ui.nav_panel("Reliability Indices", 
                    ui.output_plot("indices_plot"),
                    ui.div(ui.output_text("text_indices_analysis"), class_="insight-box")
                ),
                ui.nav_panel("λ-Var Heatmap", 
                    ui.output_plot("heatmap_plot"),
                    ui.div(ui.output_text("text_heatmap_analysis"), class_="insight-box")
                )
            ),
            value="tab_rel"
        ),
        id="main_tabs"
    ),
    title="Open Transit Delhi Data Analytics"
)


# ==========================================
# 3. SERVER LOGIC
# ==========================================
def server(input, output, session):

    @reactive.calc
    def get_active_route():
        if input.main_tabs() == "tab_otp":
            return input.route_otp()
        else:
            return input.route_rel()

    @reactive.calc
    def filtered_otp_df():
        df = master_otp_df.copy()
        if df.empty: return df
        route_val = get_active_route()
        df = df[df["route_id"].astype(str).isin(SCHEDULED_ROUTES)]
        if route_val != "All Routes":
            df = df[df["route_id"].astype(str) == route_val]
            
        df["on_time"] = df["sched_deviation"].between(-1, 5)
        df["COV_hour"] = df.groupby(["day", "hour"], observed=False)["real_headway"].transform(lambda x: x.std() / x.mean())
        df["waiting_time"] = 0.5 * df["real_headway"] * (1 + df["COV_hour"]**2)
        return df

    @reactive.calc
    def filtered_raw_df():
        df = master_raw_df.copy()
        if df.empty: return df
        route_val = get_active_route()
        df = df[df["route_id"].astype(str).isin(RELIABILITY_ROUTES)]
        if route_val != "All Routes":
            df = df[df["route_id"].astype(str) == route_val]
            
        return df

    # --- KPI Text Outputs ---
    @render.text
    def kpi_otp():
        df = filtered_otp_df()
        return f"{(df['on_time'].mean() * 100):.1f}%" if not df.empty and "on_time" in df.columns else "0%"

    @render.text
    def kpi_dev():
        df = filtered_otp_df()
        return f"{df['sched_deviation'].mean():.1f} min" if not df.empty and "sched_deviation" in df.columns else "0 min"

    @render.text
    def kpi_headway():
        df = filtered_otp_df()
        return f"{df['real_headway'].mean():.1f} min" if not df.empty and "real_headway" in df.columns else "0 min"

    @render.text
    def kpi_wait():
        df = filtered_otp_df()
        return f"{df['waiting_time'].mean():.1f} min" if not df.empty and "waiting_time" in df.columns else "0 min"

    # --- Dynamic Commentary Generators ---
    @render.text
    def text_otp_analysis():
        df = filtered_otp_df()
        if df.empty: return "No data available."
        
        otp_day = df.groupby("day", observed=False)["on_time"].mean()
        otp_slot = df.groupby("time_slot", observed=False)["on_time"].mean()
        
        best_day = otp_day.idxmax()
        worst_day = otp_day.idxmin()
        worst_slot = otp_slot.idxmin()
        
        return f"📊 Conclusion: Route performance peaks on {best_day} with the highest punctuality score. Conversely, service drops lowest on {worst_day}. Operational focus is highly recommended during the '{worst_slot}' block, which presents the highest localized delays."

    @render.text
    def text_wait_analysis():
        df = filtered_otp_df()
        if df.empty: return "No data available."
        
        wait_slot = df.groupby("time_slot", observed=False)["waiting_time"].mean()
        max_wait_slot = wait_slot.idxmax()
        max_wait_val = wait_slot.max()
        
        return f"⏳ Conclusion: Passenger wait times are significantly extended during the '{max_wait_slot}' period, reaching an average of {max_wait_val:.1f} minutes. This is heavily driven by poor headway formatting or clumped vehicle dispatches."

    @render.text
    def text_interactive_trends():
        df = filtered_raw_df()
        if df.empty: return "No data available."
        
        avg_tt = df["travel_time_min"].mean()
        max_hour = df.groupby("hour")["travel_time_min"].mean().idxmax()
        
        return f"📈 Conclusion: Across all monitored runs, the baseline average travel time settles around {avg_tt:.1f} minutes. Congestion bottlenecks are prominently concentrated around hour {max_hour}:00, visible via the wide scatter dispersion and shifting boxplot distributions."

    @render.text
    def text_indices_analysis():
        df = filtered_raw_df()
        if df.empty: return "No data available."
        
        day_stats = df.groupby("day", observed=False)["travel_time_min"].agg(
            avg_tt="mean", 
            p10=lambda x: np.percentile(x.dropna(), 10) if len(x)>0 else 1
        ).reset_index()
        
        day_stats["TTI"] = day_stats["avg_tt"] / day_stats["p10"]
        worst_idx = day_stats["TTI"].idxmax()
        worst_day = day_stats.loc[worst_idx, "day"]
        worst_tti = day_stats.loc[worst_idx, "TTI"]
        
        return f"🏁 Conclusion: Commuter scheduling reliability varies across the week. {worst_day} exhibits the highest Travel Time Index (TTI) of {worst_tti:.2f}, meaning travelers must budget over {int((worst_tti-1)*100)}% extra buffer time compared to a free-flowing run."

    @render.text
    def text_heatmap_analysis():
        df = filtered_raw_df()
        if df.empty: return "No data available."
        
        var_stats = df.groupby(["day", "hour"], observed=False)["travel_time_min"].var().reset_index()
        if var_stats.empty or var_stats["travel_time_min"].isnull().all():
            return "Conclusion: Variance levels remain uniformly distributed across schedule parameters."
            
        worst_row = var_stats.loc[var_stats["travel_time_min"].idxmax()]
        
        return f"🔥 Conclusion: The λ-Var heatmap highlights critical system instability on {worst_row['day']} at hour {int(worst_row['hour'])}:00. High variance indices indicate unpredictable traffic or unstable block dispatch setups."

    # --- Static Plots ---
    @render.plot
    def plot_otp_day():
        df = filtered_otp_df()
        if df.empty or "on_time" not in df.columns: return plt.figure()
        otp_day = df.groupby("day", observed=False)["on_time"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(otp_day["day"], otp_day["on_time"] * 100, color=COLORS["blue"])
        ax.bar_label(bars, fmt='%.1f%%', padding=4)
        ax.set_title("Daywise On-time Performance")
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("On-Time Performance (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    @render.plot
    def plot_otp_slot():
        df = filtered_otp_df()
        if df.empty or "on_time" not in df.columns: return plt.figure()
        otp_slot = df.groupby("time_slot", observed=False)["on_time"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(otp_slot["time_slot"], otp_slot["on_time"] * 100, color=COLORS["orange"])
        ax.bar_label(bars, fmt='%.1f%%', padding=4)
        ax.set_title("On-Time Performance at different time periods of day")
        ax.set_xlabel("Time Periods")
        ax.set_ylabel("On-Time Performance (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    @render.plot
    def plot_wait_day():
        df = filtered_otp_df()
        if df.empty or "waiting_time" not in df.columns: return plt.figure()
        wait_day = df.groupby("day", observed=False)["waiting_time"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(wait_day["day"], wait_day["waiting_time"], color=COLORS["red"])
        ax.bar_label(bars, fmt='%.1f', padding=4)
        ax.set_title("Estimated Avg. Waiting Time for days of week")
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Avg. Waiting time (min)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    @render.plot
    def plot_wait_slot():
        df = filtered_otp_df()
        if df.empty or "waiting_time" not in df.columns: return plt.figure()
        wait_slot = df.groupby("time_slot", observed=False)["waiting_time"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(wait_slot["time_slot"], wait_slot["waiting_time"], color=COLORS["purple"])
        ax.bar_label(bars, fmt='%.1f', padding=4)
        ax.set_title("Estimated Avg. Waiting Time for different time periods of day")
        ax.set_xlabel("Time Periods")
        ax.set_ylabel("Avg. waiting time (min)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    @render.plot
    def plot_cov_heatmap():
        df = filtered_otp_df()
        if df.empty or "real_headway" not in df.columns: return plt.figure()

        cov_hour = df.groupby(["day", "hour"], observed=False)["real_headway"].agg(
            mean="mean", std="std"
        ).reset_index()

        cov_pivot = (
            (cov_hour["std"] / cov_hour["mean"])
            .to_frame(name="COV")
            .join(cov_hour)
            .pivot(index="day", columns="hour", values="COV")
            .fillna(0)
        )
        cov_pivot = cov_pivot.reindex(day_order)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cov_pivot, cmap="mako", ax=ax, cbar_kws={'label': 'COV'})

        ax.set_title("Headway Adherence (COV) Heatmap")
        ax.set_xlabel("Hour of Day", labelpad=10)
        ax.set_ylabel("Days of Week")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
        ax.tick_params(axis='y', pad=10)

        plt.tight_layout(pad=2)
        plt.subplots_adjust(bottom=0.15)
        return fig

    # --- Interactive Plots ---
    @render_widget
    def interactive_scatter():
        df = filtered_raw_df()
        if df.empty: return go.Figure()
        
        slot_stats = df.groupby("hour")["travel_time_min"].agg(
            mean_tt="mean", 
            p10=lambda x: np.percentile(x.dropna(), 10) if len(x)>0 else np.nan, 
            p95=lambda x: np.percentile(x.dropna(), 95) if len(x)>0 else np.nan
        ).reset_index()

        fig = px.scatter(df, x="hour", y="travel_time_min", opacity=0.35, 
            hover_data=["date", "day", "time_period"],
            title=f"Travel Time Distribution: Route {get_active_route()}",
            color_discrete_sequence=[COLORS["blue"]])
        fig.add_trace(go.Scatter(x=slot_stats["hour"], y=slot_stats["mean_tt"], mode='lines', name='Mean', line=dict(color=COLORS["red"], dash='dash')))
        fig.add_trace(go.Scatter(x=slot_stats["hour"], y=slot_stats["p95"], mode='lines', name='P95', line=dict(color=COLORS["green"], dash='dot')))
        fig.add_trace(go.Scatter(x=slot_stats["hour"], y=slot_stats["p10"], mode='lines', name='P10', line=dict(color=COLORS["purple"], dash='dot')))
        fig.update_layout(template="plotly_white", hovermode="closest")
        return fig

    @render_widget
    def interactive_boxplot():
        df = filtered_raw_df()
        if df.empty: return go.Figure()
        
        fig = px.box(df, x="day", y="travel_time_min", color="time_period",
            title=f"Travel Time Variation: Route {get_active_route()}",
            category_orders={"day": day_order, "time_period": slot_order})
        fig.update_layout(template="plotly_white")
        return fig

    # --- Reliability Plots ---
    @render.plot
    def indices_plot():
        df = filtered_raw_df()
        if df.empty: return plt.subplots()[0]

        day_stats = df.groupby("day", observed=False)["travel_time_min"].agg(
            avg_tt="mean", 
            p10=lambda x: np.percentile(x.dropna(), 10) if len(x)>0 else np.nan, 
            p95=lambda x: np.percentile(x.dropna(), 95) if len(x)>0 else np.nan
        ).reset_index()
    
        day_stats["p10"] = day_stats["p10"].replace(0, np.nan)
        day_stats["TTI"] = day_stats["avg_tt"] / day_stats["p10"]
        day_stats["PTI"] = day_stats["p95"] / day_stats["p10"]
        day_stats["BTI"] = ((day_stats["p95"] - day_stats["avg_tt"]) / day_stats["avg_tt"]) * 100

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(day_stats["day"], day_stats["TTI"], marker="o", color=COLORS["blue"], label="TTI")
        ax1.plot(day_stats["day"], day_stats["PTI"], marker="s", color=COLORS["orange"], label="PTI")
        ax1.set_ylabel("TTI / PTI", color=COLORS["blue"])
        ax1.tick_params(axis='y', labelcolor=COLORS["blue"])

        ax2 = ax1.twinx()
        ax2.plot(day_stats["day"], day_stats["BTI"], marker="^", color=COLORS["green"], label="BTI (%)")
        ax2.set_ylabel("BTI (%)", color=COLORS["green"], fontsize=11, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=COLORS["green"], width=2)
        ax2.set_ylim(0, 100)
        
        ax1.set_title("Reliability Indices vs Day of Week")
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_linewidth(2)
        plt.tight_layout()
        return fig

    @render.plot
    def heatmap_plot():
        df = filtered_raw_df()
        if df.empty: return plt.figure()
        
        pivot_data = df.groupby(["day", "hour"], observed=False)["travel_time_min"].var().unstack().fillna(0)
        pivot_data = pivot_data.reindex(day_order)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot_data, cmap="rocket", ax=ax, cbar_kws={'label': 'Variance ($\sigma^2$)'})
        ax.set_title("Travel Travel Variance ($\lambda$-Var) Heatmap")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")
        plt.tight_layout()
        return fig

app = App(app_ui, server)
