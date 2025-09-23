# app.py (patched: safer ocean handling + minor cleanups)
import json
import os
import glob
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from flask_caching import Cache
import geopandas as gpd
import xarray as xr
from data_processing import compute_spatial_mean_timeseries, compute_spatial_map_mean, seasonally_aggregate
from datetime import datetime
# Import data processing functions (from updated data_processing.py)
from data_processing import (
    process_and_clip_all_scenarios,
    analyze_data,
    get_indonesia_regencies,
    process_and_clip_all_scenarios_extreme,
    analyze_extreme_data,
    extreme_variables_config,
    process_and_clip_all_scenarios_extreme,
    GDF_BATAS
)

def hex_to_rgba(hex_color, alpha=1.0):
    try:
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'
    except:
        return f'rgba(0, 0, 0, {alpha})'

# --- Dash App Initialization ---
app = dash.Dash(
    __name__,
    external_stylesheets=[
        'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'
    ],
    title="Dashboard Analisis Iklim Indonesia"
)
server = app.server

# --- Cache ---
CACHE_CONFIG = {
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_DEFAULT_TIMEOUT': 3600 * 24
}
cache = Cache(app.server, config=CACHE_CONFIG)

@cache.memoize(timeout=3600 * 24)
def memoized_process_and_clip_all_scenarios(aoi_name, var_name):
    return process_and_clip_all_scenarios(aoi_name, var_name)

@cache.memoize(timeout=3600 * 24)
def memoized_analyze_data(clipped_data):
    # analyze_data expects descriptor; our memoization wrapper here purposely re-calls analyze_data
    return analyze_data(clipped_data)

@cache.memoize(timeout=3600 * 24)
def memoized_process_and_clip_all_scenarios_extreme(aoi_name, var_name):
    return process_and_clip_all_scenarios_extreme(aoi_name, var_name)

@cache.memoize(timeout=3600 * 24)
def memoized_analyze_extreme_data(clipped_data):
    # analyze_data expects descriptor; our memoization wrapper here purposely re-calls analyze_data
    return analyze_extreme_data(clipped_data)

# --- Geodata ---
list_regencies = get_indonesia_regencies()

# --- Variabel Data ---
data_variables_config = {
    'pr': {'label': 'Curah Hujan', 'unit': 'mm'},
    'tas': {'label': 'Suhu Rata-rata', 'unit': '°C'},
    'tasmin': {'label': 'Suhu Minimum', 'unit': '°C'},
    'tasmax': {'label': 'Suhu Maksimum', 'unit': '°C'}
}

extreme_variables = {
    "cdd": "Consecutive Dry Days (CDD)",
    "cwd": "Consecutive Wet Days (CWD)",
    "rx1day": "RX1day - Highest 1-day precipitation",
    "rx5day": "RX5day - Highest 5-day precipitation",
    "r20mm": "R20mm - Very heavy precipitation days",
}

plot_options = [
    {'label': 'Klimatologi Bulanan', 'value': 'clim'},
    {'label': 'Tren Tahunan', 'value': 'trend'},
    {'label': 'Perubahan Persentase', 'value': 'change'},
    {'label': 'Spasial', 'value': 'spatial'}
]

DECADES = [
    ('2021-2050', '2021', '2050'),
    ('2031-2060', '2031', '2060'),
    ('2041-2070', '2041', '2070'),
    ('2051-2080', '2051', '2080'),
    ('2061-2090', '2061', '2090'),
    ('2071-2100', '2071', '2100'),
]

SCENARIO_COLORS = {
    'ssp126': '#00FF00',
    'ssp245': '#0000FF',
    'ssp370': '#FFA500',
    'ssp585': '#FF0000'
}

# --- Layout (ke samaan UI Anda) ---
app.layout = html.Div(children=[
    html.Div(className='header', children=[
        html.Img(src='https://www.ipb.ac.id/wp-content/uploads/2023/12/Logo-IPB-University_Horizontal-Putih.png', height='50px', style={'marginRight': '12px'}),
        html.H1("Dashboard Analisis Iklim Indonesia"),
        html.Div(style={'width': '50px'})
    ]),

    html.Div(className='body-content', children=[
        dcc.Tabs(id="main-tabs", value='tab-1', children=[
            dcc.Tab(label='Pilih Domain Data', value='tab-1', children=[
                html.Div(className='tab-content', children=[
                    html.H2("Peta Kabupaten/Kota Indonesia"),
                    html.Label("Pilih Kabupaten/Kota:"),
                    dcc.Dropdown(
                        id='regency-dropdown',
                        options=[{'label': reg, 'value': reg} for reg in list_regencies],
                        placeholder='Pilih Kabupaten/Kota...'
                    ),
                    dcc.Graph(
                        id='indonesia-map',
                        config={'scrollZoom': True},
                        className='dccGraphContainer'
                    ),
                    html.Div(id='map-info-container', className='map-info-container', children=[
                        html.Label("Kabupaten/Kota Terpilih:"),
                        html.Div(id='selected-regency-output')
                    ])
                ])
            ]),
            dcc.Tab(label='Suhu Udara dan Curah Hujan', value='tab-2', children=[
                html.Div(className='tab-content', children=[
                    html.H2("Visualisasi Suhu Udara dan Curah Hujan"),
                    html.Label("Pilih Variabel:"),
                    dcc.Dropdown(
                        id='main-variable-dropdown',
                        options=[{'label': cfg['label'], 'value': key} for key, cfg in data_variables_config.items()],
                        value='pr',
                        clearable=False
                    ),
                    html.Label("Pilih Jenis Plot:"),
                    dcc.Dropdown(
                        id='plot-type-dropdown',
                        options=plot_options,
                        value='clim',
                        clearable=False
                    ),
                    # Dropdown periode spasial (selalu ada tapi disembunyikan dulu)
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="spatial-period-dropdown",
                                options=[
                                    {"label": "Historis (1991–2014)", "value": "historis"},
                                    {"label": "2021–2050", "value": "2021-2050"},
                                    {"label": "2031–2060", "value": "2031-2060"},
                                    {"label": "2041–2070", "value": "2041-2070"},
                                    {"label": "2051–2080", "value": "2051-2080"},
                                    {"label": "2061–2090", "value": "2061-2090"},
                                    {"label": "2071–2100", "value": "2071-2100"},
                                ],
                                value="historis",
                                clearable=False,
                                style={"width": "60%"},
                            )
                        ],
                        id="spatial-period-container",
                        style={"display": "none"},
                    ),
                    dcc.Loading(
                        id="loading-output",
                        type="default",
                        children=html.Div(id='visualization-container')
                    )
                ])
            ]),
            dcc.Tab(label='Iklim Ekstrem', value='tab-extreme', children=[
                html.Div(className='tab-content', children=[
                    html.H2("Visualisasi Iklim Ekstrem"),
                    html.Label("Pilih Variabel Ekstrem:"),
                    dcc.Dropdown(
                        id='extreme-variable-dropdown',
                        options=[{'label': cfg['label'], 'value': key} for key, cfg in extreme_variables_config.items()],
                        value=list(extreme_variables_config.keys())[0] if len(extreme_variables_config)>0 else None,
                        clearable=False
                    ),
                    html.Label("Pilih Jenis Plot:"),
                    dcc.Dropdown(
                        id='extreme-plot-type-dropdown',
                        options=[
                            {'label': 'Tren Tahunan', 'value': 'trend'},
                            {'label': 'Spasial', 'value': 'spatial'},
                        ],
                        value='trend',
                        clearable=False
                    ),
                    # dropdown periode untuk tab-extreme (selalu ada tapi tersembunyi awalnya)
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="extreme-spatial-period-dropdown",
                                options=[
                                    {"label": "Historis (1991–2014)", "value": "historis"},
                                    {"label": "2021–2050", "value": "2021-2050"},
                                    {"label": "2031–2060", "value": "2031-2060"},
                                    {"label": "2041–2070", "value": "2041-2070"},
                                    {"label": "2051–2080", "value": "2051-2080"},
                                    {"label": "2061–2090", "value": "2061-2090"},
                                    {"label": "2071–2100", "value": "2071-2100"},
                                ],
                                value="historis",
                                clearable=False,
                                style={"width": "60%"},
                            )
                        ],
                        id="extreme-spatial-period-container",
                        style={"display": "none", "marginTop": "8px"},
                    ),
                    dcc.Loading(id="loading-extreme-output", type="default",
                                children=html.Div(id='extreme-visualization-container')),
                ])
            ]),
            dcc.Tab(label='Suhu Permukaan Lautan', value='tab-ocean', children=[
                html.Div(className='tab-content', children=[
                    html.H2("Visualisasi Suhu Permukaan Laut"),
                    dcc.ConfirmDialog(
                        id='ocean-popup',
                        message="Tab ini sedang dalam pengembangan."
                    ),
                    html.Div(id='ocean-dummy-container')  # dummy output
                ])
            ]),
        ])
    ], style={'maxWidth': '1400px', 'margin': '0 auto'}),

    dcc.Download(id='download-data'),

    html.Div(className='footer', children=[
        html.P("© 2025 Dashboard Analisis Iklim Indonesia")
    ])
])

# --- Map callbacks (unchanged logic) ---
@app.callback(
    Output('indonesia-map', 'figure'),
    Input('regency-dropdown', 'value')
)
def update_map(selected_regency):
    shp_path = "assets/shp/gadm41_IDN_2.shp"
    gdf = gpd.read_file(shp_path)

    if gdf.crs and gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs(epsg=3857)
        centroids = gdf_proj.centroid.to_crs(epsg=4326)
        gdf['lon'] = centroids.x
        gdf['lat'] = centroids.y
    else:
        gdf['lon'] = gdf.centroid.x
        gdf['lat'] = gdf.centroid.y

    gdf = gdf.rename(columns={'NAME_2': 'regency'})
    df = pd.DataFrame(gdf[['regency', 'lon', 'lat']])

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        hover_name="regency",
        zoom=3.5,
        center={"lat": -2.0, "lon": 118.0},
        mapbox_style="carto-positron"
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8, color='blue'))

    if selected_regency:
        regency_data = df[df['regency'] == selected_regency]
        if not regency_data.empty:
            fig.update_layout(
                mapbox_center={"lat": regency_data['lat'].iloc[0], "lon": regency_data['lon'].iloc[0]},
                mapbox_zoom=8
            )
            fig.add_trace(go.Scattermapbox(
                lat=[regency_data['lat'].iloc[0]],
                lon=[regency_data['lon'].iloc[0]],
                mode='markers',
                marker=go.scattermapbox.Marker(size=12, color='red'),
                hoverinfo='none',
                showlegend=False
            ))

    fig.update_layout(title="Peta Kabupaten/Kota Indonesia", title_x=0.5, height=600, margin=dict(t=60))
    return fig

@app.callback(
    Output('regency-dropdown', 'value'),
    Input('indonesia-map', 'clickData'),
    State('regency-dropdown', 'value')
)
def update_dropdown_on_click(clickData, current_value):
    if clickData and 'points' in clickData:
        pt = clickData['points'][0]
        selected_regency = pt.get('hovertext') or pt.get('text') or pt.get('hoverlabel')
        return selected_regency
    return current_value

@app.callback(
    Output('selected-regency-output', 'children'),
    Input('regency-dropdown', 'value')
)
def update_selected_regency(selected_regency):
    if not selected_regency:
        return "Belum ada Kabupaten/Kota terpilih"
    return f"Kabupaten/Kota yang dipilih: {selected_regency}"

@app.callback(
    Output("spatial-period-container", "style"),
    Input("plot-type-dropdown", "value")
)
def toggle_spatial_period_dropdown(selected_plot):
    if selected_plot == "spatial":
        return {"display": "block"}
    return {"display": "none"}

@app.callback(
    Output("extreme-spatial-period-container", "style"),
    Input("extreme-plot-type-dropdown", "value"),
)
def toggle_extreme_spatial_dropdown(plot_type):
    if plot_type == "spatial":
        return {"display": "block"}
    return {"display": "none"}

# ====== REQUIREMENTS: place this BEFORE def update_visualization ======
import matplotlib
matplotlib.use("Agg")
from data_processing import GDF_BATAS  # GeoDataFrame batas wilayah
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import io, base64

@app.callback(
    Output('visualization-container', 'children'),
    Input('main-tabs', 'value'),
    Input('main-variable-dropdown', 'value'),
    Input('plot-type-dropdown', 'value'),
    Input('selected-regency-output', 'children'),
    Input('spatial-period-dropdown', 'value')
)
def update_visualization(selected_tab, selected_var,
                         selected_plot, selected_regency_text,
                         spatial_period=None):
    # Pastikan AOI valid
    if "Kabupaten/Kota yang dipilih: " in selected_regency_text:
        aoi = selected_regency_text.replace("Kabupaten/Kota yang dipilih: ", "").strip()
    else:
        return html.Div("Pilih Kabupaten/Kota terlebih dahulu.", style={'color': 'red'})

    if selected_tab != "tab-2":
        return dash.no_update

    print("CALLBACK TAB-2 DIPANGGIL:", selected_var, selected_plot, spatial_period)

    try:
        descriptor = memoized_process_and_clip_all_scenarios(aoi, selected_var)
        processed = memoized_analyze_data(descriptor)
    except Exception as e:
        return html.Div(f"Error memuat data: {e}", style={'color': 'red'})

    var_label = data_variables_config[selected_var]['label']
    var_unit = data_variables_config[selected_var]['unit']

    # ambil data hasil analisis
    mclim_hist = processed.get('mclim_historis', pd.DataFrame())
    mclim_proy_dec = processed.get('mclim_proyeksi_decade', {})
    pct_change_dec = processed.get('percent_change_decade', {})
    annual_stats = processed.get('annual_stats_combined', {})
    clipped_arrays = processed.get('clipped_arrays', {})

    # ---------------- Spatial (pakai Matplotlib)
    if selected_plot == "spatial":
        def plot_one_map(ax, da, title, var_label, var_unit, vmin=None, vmax=None):
            lons = da["lon"].values
            lats = da["lat"].values
            z = da.values
            im = ax.pcolormesh(
                lons, lats, z,
                cmap="viridis", shading="auto",
                vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree()
            )
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.coastlines(resolution="110m", linewidth=0.5)
            gl = ax.gridlines(
                draw_labels=True, linewidth=0.1, linestyle="--", color="gray"
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 7}
            gl.ylabel_style = {"size": 7}
            if not GDF_BATAS.empty:
                GDF_BATAS.boundary.plot(ax=ax, linewidth=0.4, edgecolor="black")

            pad_x = (lons.max()-lons.min())*0.5
            pad_y = (lats.max()-lats.min())*0.5
            ax.set_extent([lons.min()-pad_x, lons.max()+pad_x,
                            lats.min()-pad_y, lats.max()+pad_y],
                            crs=ccrs.PlateCarree())
            # paksa kotak
            ax.set_aspect('auto')
            return im

        if spatial_period == "historis":
            da_hist = clipped_arrays.get("historis")
            if da_hist is None:
                return html.Div("Data historis tidak tersedia", style={"color":"red"})
            grouped = da_hist.groupby("TIME.month").mean("TIME")
            months = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul",
                        "Agu","Sep","Okt","Nov","Des"]
            fig, axes = plt.subplots(4,3, figsize=(10, 10), dpi=100,
                                        subplot_kw={"projection":ccrs.PlateCarree()})
            fig.subplots_adjust(wspace=0.15, hspace=0.25, bottom=0.15)
            vmin = float(grouped.min().values)
            vmax = float(grouped.max().values)
            for i, month in enumerate(months, start=1):
                r, c = divmod(i-1, 3)
                da_m = grouped.sel(month=i)
                im = plot_one_map(axes[r,c], da_m, month, var_label, var_unit,
                                    vmin=vmin, vmax=vmax)
            cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            cbar.set_label(f"{var_label} ({var_unit})", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            fig.suptitle(f"Rata-Rata {var_label} {aoi} 1991–2014",
                            fontsize=10, fontweight="bold", y=0.95)
            buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
            img_src = "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()
            return html.Div([html.Img(src=img_src,
                                        style={"display":"block","margin":"0 auto",
                                                "width":"75%","height":"auto"})])

        # --- Proyeksi per periode (4 gambar, masing-masing 12 subplot bulanan)
        try:
            ystart, yend = map(int, spatial_period.split("-"))
        except Exception:
            return html.Div("Periode tidak valid", style={"color":"red"})

        plot_divs = []
        months = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul",
                "Agu","Sep","Okt","Nov","Des"]
        for scn in ["ssp126","ssp245","ssp370","ssp585"]:
            da = clipped_arrays.get(scn)
            if da is None:
                continue

            # filter sesuai periode
            mask = (da["TIME"].dt.year >= ystart) & (da["TIME"].dt.year <= yend)
            da_sel = da.sel(TIME=mask)
            grouped = da_sel.groupby("TIME.month").mean("TIME")

            # tentukan skala global agar konsisten antar subplot
            vmin = float(grouped.min().values)
            vmax = float(grouped.max().values)

            # buat 12 subplot bulanan
            fig, axes = plt.subplots(4, 3, figsize=(10, 10), dpi=100,
                                    subplot_kw={"projection": ccrs.PlateCarree()})
            fig.subplots_adjust(wspace=0.15, hspace=0.25, bottom=0.15)

            for i, month in enumerate(months, start=1):
                r, c = divmod(i-1, 3)
                da_m = grouped.sel(month=i)
                im = plot_one_map(axes[r, c], da_m, month, var_label, var_unit,
                                vmin=vmin, vmax=vmax)

            # tambahkan colorbar
            cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            cbar.set_label(f"{var_label} ({var_unit})", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

            # judul utama
            fig.suptitle(f"Rata-Rata {var_label} {aoi} {ystart}-{yend} ({scn.upper()})",
                        fontsize=10, fontweight="bold", y=0.95)

            # simpan ke buffer → base64
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            img_src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

            # simpan ke plot_divs
            plot_divs.append(
                html.Div([
                    html.Img(
                        src=img_src,
                        style={
                            "display": "block",
                            "margin": "0 auto",
                            "width": "95%",
                            "height": "auto"
                        }
                    )
                ])
            )

        if not plot_divs:
            return html.Div("Tidak ada data proyeksi untuk periode ini.", style={"color": "red"})

        # tata jadi 2x2 grid
        return html.Div(
            children=plot_divs,
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",  # 2 kolom
                "gap": "20px",
                "justifyItems": "center",
                "alignItems": "center"
            }
        )

    # ---------------------------
    # KLIMATOLOGI BULANAN (tidak diubah logikanya — hanya wrapper)
    # ---------------------------
    elif selected_plot == 'clim':
        plot_divs = []
        for (label, ystart, yend) in DECADES:
            fig = go.Figure()
            decade_key = f"{ystart}-{yend}"
            # safety check: pastikan mclim_hist benar-benar DataFrame sebelum akses .empty/.columns
            if isinstance(mclim_hist, pd.DataFrame) and (not mclim_hist.empty) and ('mean' in mclim_hist.columns):
                fig.add_trace(go.Scatter(x=mclim_hist['month'], y=mclim_hist['mean'],
                                        mode='lines+markers',
                                        line=dict(color='black', width=3),
                                        name='Historis (1991–2014)', marker=dict(size=8)))
            dec_dict = mclim_proy_dec.get(decade_key, {})
            for scn in ['ssp126','ssp245','ssp370','ssp585']:
                df_scn = dec_dict.get(scn)
                if df_scn is None or df_scn.empty or 'mean' not in df_scn.columns:
                    continue
                fig.add_trace(go.Scatter(x=df_scn['month'], y=df_scn['mean'],
                                        mode='lines+markers',
                                        line=dict(color=SCENARIO_COLORS[scn], width=2),
                                        name=scn.upper(), marker=dict(size=5)))
            fig.update_layout(title=f'Klimatologi Bulanan {var_label}<br>{aoi} ({label})',
                            xaxis=dict(tickvals=list(range(1,13)),
                                        ticktext=['Jan','Feb','Mar','Apr','Mei','Jun',
                                                'Jul','Agu','Sep','Okt','Nov','Des'],
                                        title="Bulan"),
                            yaxis=dict(title=f'{var_label} ({var_unit})'),
                            title_font=dict(size=12), margin=dict(t=60, b=40),
                            legend=dict(orientation='h', yanchor='top',
                                        y=-0.25, x=0.5, xanchor='center'))
            plot_divs.append(html.Div([
                dcc.Graph(figure=fig, config={'responsive': True}, style={'width': '100%', 'height': '480px'}),
                dbc.Button("Download CSV",
                        id={"type": "download-btn", "scope": "clim", "period": decade_key},
                        color="primary", size="sm", className="mt-2")
            ], style={'width': '100%', 'display': 'block'}))
        return html.Div(plot_divs, className='visualization-grid-2x2')

    # ---------------------------
    # TREN TAHUNAN (sama)
    # ---------------------------
    elif selected_plot == 'trend':
        fig = go.Figure()
        hist_df = annual_stats.get('historis', pd.DataFrame())
        if not hist_df.empty:
            fig.add_trace(go.Scatter(x=hist_df['year'], y=hist_df['mean'],
                                    mode='lines', line=dict(color='black', width=3),
                                    name='Historis (1991–2014)'))
        for scn in ['ssp126','ssp245','ssp370','ssp585']:
            df_sc = annual_stats.get(scn, pd.DataFrame())
            if df_sc.empty:
                continue
            fig.add_trace(go.Scatter(x=df_sc['year'], y=df_sc['min'],
                                    mode='lines',
                                    line=dict(color='rgba(0,0,0,0)', width=0),
                                    showlegend=False))
            fig.add_trace(go.Scatter(x=df_sc['year'], y=df_sc['max'],
                                    mode='lines', fill='tonexty',
                                    fillcolor=hex_to_rgba(SCENARIO_COLORS[scn], 0.1),
                                    line=dict(color='rgba(0,0,0,0)', width=0),
                                    showlegend=False))
            fig.add_trace(go.Scatter(x=df_sc['year'], y=df_sc['mean'],
                                    mode='lines',
                                    line=dict(color=SCENARIO_COLORS[scn], width=2),
                                    name=scn.upper()))
        fig.update_layout(title=f'Tren Tahunan {var_label} ({aoi}) [1991–2100]',
                        xaxis_title='Tahun', yaxis_title=f'{var_label} ({var_unit})',
                        title_font=dict(size=12), margin=dict(t=70, b=40))
        return html.Div([
            dcc.Graph(figure=fig, config={'responsive': True}, style={'width': '100%', 'height': '480px'}),
            dbc.Button("Download CSV",
                    id={"type": "download-btn", "scope": "trend", "period": "all"},
                    color="primary", size="sm", className="mt-2")
        ], style={'width': '100%', 'display': 'block'})

    # ---------------------------
    # PERUBAHAN PERSENTASE (tidak diubah logikanya)
    # ---------------------------
    elif selected_plot == 'change':
        plot_divs = []
        for (label, ystart, yend) in DECADES:
            fig = go.Figure()
            decade_key = f"{ystart}-{yend}"
            dec_dict = pct_change_dec.get(decade_key, {})
            for scn in ['ssp126','ssp245','ssp370','ssp585']:
                df_scn = dec_dict.get(scn)
                if df_scn is None or df_scn.empty or 'pct' not in df_scn.columns:
                    continue
                fig.add_trace(go.Bar(x=df_scn['month'], y=df_scn['pct'],
                                    name=scn.upper(), marker_color=SCENARIO_COLORS[scn]))
            fig.update_layout(title=f'Perubahan Persentase {var_label}<br>{aoi} ({label})',
                            xaxis=dict(tickvals=list(range(1,13)),
                                        ticktext=['Jan','Feb','Mar','Apr','Mei','Jun',
                                                'Jul','Agu','Sep','Okt','Nov','Des'],
                                        title='Bulan'),
                            yaxis=dict(title='% Perubahan'),
                            title_font=dict(size=12), margin=dict(t=60, b=40),
                            barmode='group',
                            legend=dict(orientation='h', yanchor='top',
                                        y=-0.25, x=0.5, xanchor='center'))
            plot_divs.append(html.Div([
                dcc.Graph(figure=fig, config={'responsive': True}, style={'width': '100%', 'height': '480px'}),
                dbc.Button("Download CSV",
                        id={"type": "download-btn", "scope": "change", "period": decade_key},
                        color="primary", size="sm", className="mt-2")
            ], style={'width': '100%', 'display': 'block'}))
        return html.Div(plot_divs, className='visualization-grid-2x3')

@app.callback(
    Output('extreme-visualization-container', 'children'),
    Input('extreme-variable-dropdown', 'value'),
    Input('extreme-plot-type-dropdown', 'value'),
    Input('extreme-spatial-period-dropdown', 'value'),
    Input('selected-regency-output', 'children')
)
def update_extreme(extreme_var, selected_plot, spatial_period, selected_regency_text):
        # Pastikan AOI valid
    if "Kabupaten/Kota yang dipilih: " in selected_regency_text:
        aoi = selected_regency_text.replace("Kabupaten/Kota yang dipilih: ", "").strip()
    else:
        return html.Div("Pilih Kabupaten/Kota terlebih dahulu.", style={'color': 'red'})
    print("CALLBACK EXTREME DIPANGGIL:", extreme_var, selected_plot, spatial_period)

    try:
        clipped_ext = memoized_process_and_clip_all_scenarios_extreme(aoi, extreme_var)
        processed_ext = memoized_analyze_extreme_data(clipped_ext)
    except Exception as e:
        return html.Div(f"Error memuat data ekstrem: {e}", style={"color": "red"})

    var_label = extreme_variables_config[extreme_var]["label"]
    var_unit = extreme_variables_config[extreme_var]["unit"]

    # ambil hasil analisis
    annual_stats_ext = processed_ext.get("annual_stats_combined", {})
    clipped_arrays_ext = clipped_ext

    print("DEBUG annual_stats_ext keys:", annual_stats_ext.keys())
  
    if selected_plot == 'trend':
        clipped_ext = memoized_process_and_clip_all_scenarios_extreme(aoi, extreme_var)
        processed_ext = memoized_analyze_extreme_data(clipped_ext)
        fig = go.Figure()
        hist_df = annual_stats_ext.get('historis', pd.DataFrame())
        if not hist_df.empty:
            fig.add_trace(go.Scatter(x=hist_df['year'], y=hist_df['mean'],
                                    mode='lines', line=dict(color='black', width=3),
                                    name='Historis (1991–2014)'))
        for scn in ['ssp126','ssp245','ssp370','ssp585']:
            df_sc = annual_stats_ext.get(scn, pd.DataFrame())
            if df_sc.empty:
                continue
            fig.add_trace(go.Scatter(x=df_sc['year'], y=df_sc['min'],
                                    mode='lines',
                                    line=dict(color='rgba(0,0,0,0)', width=0),
                                    showlegend=False))
            fig.add_trace(go.Scatter(x=df_sc['year'], y=df_sc['max'],
                                    mode='lines', fill='tonexty',
                                    fillcolor=hex_to_rgba(SCENARIO_COLORS[scn], 0.1),
                                    line=dict(color='rgba(0,0,0,0)', width=0),
                                    showlegend=False))
            fig.add_trace(go.Scatter(x=df_sc['year'], y=df_sc['mean'],
                                    mode='lines',
                                    line=dict(color=SCENARIO_COLORS[scn], width=2),
                                    name=scn.upper()))
        fig.update_layout(title=f'Tren Tahunan {var_label} ({aoi}) [1991–2100]',
                        xaxis_title='Tahun', yaxis_title=f'{var_label} ({var_unit})',
                        title_font=dict(size=12), margin=dict(t=70, b=40))
        return html.Div([
            dcc.Graph(figure=fig, config={'responsive': True}, style={'width': '100%', 'height': '480px'}),
            dbc.Button("Download CSV",
                    id={"type": "download-btn", "scope": "extreme_trend", "period": "all"},
                    color="primary", size="sm", className="mt-2")
        ], style={'width': '100%', 'display': 'block'})

    elif selected_plot == "spatial":
        def plot_one_map(ax, da, title, var_label, var_unit, vmin=None, vmax=None):
            lons = da["lon"].values
            lats = da["lat"].values
            z = da.values
            im = ax.pcolormesh(
                lons, lats, z,
                cmap="viridis", shading="auto",
                vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree()
            )
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.coastlines(resolution="110m", linewidth=0.5)
            gl = ax.gridlines(
                draw_labels=True, linewidth=0.1, linestyle="--", color="gray"
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 7}
            gl.ylabel_style = {"size": 7}
            if not GDF_BATAS.empty:
                GDF_BATAS.boundary.plot(ax=ax, linewidth=0.4, edgecolor="black")

            pad_x = (lons.max()-lons.min())*0.5
            pad_y = (lats.max()-lats.min())*0.5
            ax.set_extent([lons.min()-pad_x, lons.max()+pad_x,
                            lats.min()-pad_y, lats.max()+pad_y],
                            crs=ccrs.PlateCarree())
            # paksa kotak
            ax.set_aspect('auto')
            return im

        if spatial_period=="historis":
            da_hist=clipped_arrays_ext.get("historis")
            if da_hist is None: return html.Div("Data historis tidak ada", style={"color":"red"})
            da_sel=da_hist.mean("TIME")
            fig,ax=plt.subplots(1,1,figsize=(10,10),dpi=100,
                                subplot_kw={"projection":ccrs.PlateCarree()})
            fig.subplots_adjust(wspace=0.15, hspace=0.25, bottom=0.15)
            im = plot_one_map(ax, da_sel, "",
                            var_label, var_unit,
                            vmin=float(da_sel.min()), vmax=float(da_sel.max()))
            cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            cbar.set_label(f"{var_label} ({var_unit})", fontsize=14)
            cbar.ax.tick_params(labelsize=14)
            fig.suptitle(f"Rata-Rata {var_label} {aoi} 1991–2014",
                            fontsize=16, fontweight="bold", y=0.95)
            buf=io.BytesIO(); fig.savefig(buf,format="png",bbox_inches="tight"); plt.close(fig)
            img_src="data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()
            return html.Div([html.Img(src=img_src,
                                        style={"display":"block","margin":"0 auto",
                                                "width":"50%","height":"auto"})])

        # --- Proyeksi: 4 subplot skenario dalam 1 gambar
        try:
            ystart,yend=map(int,spatial_period.split("-"))
        except Exception:
            return html.Div("Periode tidak valid",style={"color":"red"})
        scenarios_data={}
        for scn in ["ssp126","ssp245","ssp370","ssp585"]:
            da=clipped_arrays_ext.get(scn)
            if da is None: continue
            mask=(da["TIME"].dt.year>=ystart)&(da["TIME"].dt.year<=yend)
            da_sel=da.sel(TIME=mask).mean("TIME")
            scenarios_data[scn]=da_sel
        vmin=np.nanmin([float(x.min().values) for x in scenarios_data.values()])
        vmax=np.nanmax([float(x.max().values) for x in scenarios_data.values()])
        fig,axes=plt.subplots(2,2,figsize=(10,10),dpi=100,
                                subplot_kw={"projection":ccrs.PlateCarree()})
        fig.subplots_adjust(wspace=0.15, hspace=0.25, bottom=0.15)
        axes=axes.ravel()
        for ax,(scn,da_sel) in zip(axes,scenarios_data.items()):
            im=plot_one_map(ax,da_sel,scn.upper(),var_label,var_unit,vmin,vmax)
        cbar_ax=fig.add_axes([0.25,0.08,0.5,0.02])
        cbar=fig.colorbar(im,cax=cbar_ax,orientation="horizontal")
        cbar.set_label(f"{var_label} ({var_unit})", fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        fig.suptitle(f"Rata-Rata {var_label} {aoi} {ystart}-{yend} ({scn.upper()}",
                        fontsize=12,fontweight="bold")
        buf=io.BytesIO(); fig.savefig(buf,format="png",bbox_inches="tight"); plt.close(fig)
        img_src="data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()
        return html.Div([html.Img(src=img_src,
                                    style={"display":"block","margin":"0 auto",
                                            "width":"75%","height":"auto"})])
    
# --- Single download callback (extend to extreme) ---
@app.callback(
    Output('download-data', 'data'),
    Input({'type': 'download-btn', 'scope': ALL, 'period': ALL}, 'n_clicks'),
    State('main-variable-dropdown', 'value'),
    State('extreme-variable-dropdown', 'value'),
    State('selected-regency-output', 'children'),
    State('plot-type-dropdown', 'value'),
    State('extreme-plot-type-dropdown', 'value'),
    prevent_initial_call=True
)
def handle_any_download(n_clicks_list,
                        selected_var, extreme_var,
                        selected_regency_text,
                        selected_plot, extreme_plot):
    """
    Handler download CSV dari tombol download per-plot.
    Sekarang mendukung tab utama (clim, trend, change, spatial)
    dan tab ekstrem (trend, spatial).
    """
    if not n_clicks_list or all(nc is None or nc == 0 for nc in n_clicks_list):
        return dash.no_update

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        triggered_id = json.loads(prop_id)
    except Exception:
        return dash.no_update

    if triggered_id.get("type") != "download-btn":
        return dash.no_update

    scope = triggered_id.get('scope')
    period = triggered_id.get('period')

    # Tentukan AOI
    if isinstance(selected_regency_text, str) and "Kabupaten/Kota yang dipilih:" in selected_regency_text:
        aoi = selected_regency_text.split("Kabupaten/Kota yang dipilih:")[-1].strip()
    else:
        aoi = selected_regency_text if isinstance(selected_regency_text, str) else None

    if not aoi:
        return dash.no_update

    # Pilih variable sesuai tab
    if scope.startswith("extreme"):
        var = extreme_var
    else:
        var = selected_var

    if not var:
        return dash.no_update

    # Nama file dasar
    timestr = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename_base = f"{aoi.replace(' ','_')}_{var}_{scope}_{period or 'all'}_{timestr}"

    try:
        # ----------------- TIMESERIES / TREND -----------------
        if scope in ('timeseries', 'ts', 'trend'):
            scenario = period if period in ('historis','ssp126','ssp245','ssp370','ssp585') else 'historis'
            df_ts = compute_spatial_mean_timeseries(aoi, selected_var, scenario)
            if df_ts is None or df_ts.empty:
                return dash.no_update
            return dcc.send_data_frame(df_ts.to_csv, f"{filename_base}.csv", index=False)

        # ----------------- CLIMATOLOGY (monthly) -----------------
        if scope in ('clim', 'monthly'):
            descriptor = memoized_process_and_clip_all_scenarios(aoi, selected_var)
            processed = memoized_analyze_data(descriptor)
            hist = processed.get('mclim_historis', pd.DataFrame())

            if period and isinstance(period, str) and '-' in period:
                proj_dict = processed.get('mclim_proyeksi_decade', {}).get(period, {})
                df_out = hist.set_index('month').rename(columns={'mean':'historis'}) if not hist.empty else pd.DataFrame()
                for scn, df_sc in proj_dict.items():
                    if df_sc is None or df_sc.empty:
                        continue
                    df_out[scn] = df_sc.set_index('month')['mean']
                df_out = df_out.reset_index()
                return dcc.send_data_frame(df_out.to_csv, f"{filename_base}.csv", index=False)
            else:
                if hist is None or hist.empty:
                    return dash.no_update
                return dcc.send_data_frame(hist.to_csv, f"{filename_base}.csv", index=False)

        # ----------------- PERCENT CHANGE -----------------
        if scope in ('change', 'percent_change'):
            if not period or '-' not in str(period):
                return dash.no_update
            descriptor = memoized_process_and_clip_all_scenarios(aoi, selected_var)
            processed = memoized_analyze_data(descriptor)
            pct_dict = processed.get('percent_change_decade', {}).get(period, {})
            if not pct_dict:
                return dash.no_update
            df_out = None
            for scn, df_sc in pct_dict.items():
                if df_sc is None or df_sc.empty:
                    continue
                if df_out is None:
                    df_out = df_sc.set_index('month').rename(columns={'pct': scn})
                else:
                    df_out[scn] = df_sc.set_index('month')['pct']
            if df_out is None:
                return dash.no_update
            return dcc.send_data_frame(df_out.reset_index().to_csv, f"{filename_base}.csv", index=False)

        # ----------------- SPATIAL (grid) -----------------
        if scope in ('spatial', 'map'):
            scenario = period if period in ('historis','ssp126','ssp245','ssp370','ssp585') else 'historis'
            descriptor = memoized_process_and_clip_all_scenarios(aoi, selected_var)
            processed = memoized_analyze_data(descriptor)
            da = processed.get('clipped_arrays', {}).get(scenario)
            if da is None:
                return dash.no_update
            try:
                da_mean = da.mean(dim='TIME') if 'TIME' in da.dims else da
            except Exception:
                da_mean = da
            lat_name = next((c for c in da_mean.coords if 'lat' in c.lower()), None)
            lon_name = next((c for c in da_mean.coords if 'lon' in c.lower()), None)
            if lat_name is None or lon_name is None:
                return dash.no_update
            lats = np.asarray(da_mean[lat_name].values)
            lons = np.asarray(da_mean[lon_name].values)
            z = np.squeeze(np.asarray(da_mean.values))
            rows = []
            try:
                if z.ndim == 2 and z.shape == (lats.size, lons.size):
                    for i, lat in enumerate(lats):
                        for j, lon in enumerate(lons):
                            rows.append({'lat': float(lat), 'lon': float(lon),
                                         'value': (float(z[i, j]) if not np.isnan(z[i, j]) else None)})
                else:
                    zz = np.reshape(z, (lats.size, lons.size))
                    for i, lat in enumerate(lats):
                        for j, lon in enumerate(lons):
                            rows.append({'lat': float(lat), 'lon': float(lon),
                                         'value': (float(zz[i, j]) if not np.isnan(zz[i, j]) else None)})
            except Exception:
                return dash.no_update
            df_out = pd.DataFrame(rows)
            return dcc.send_data_frame(df_out.to_csv, f"{filename_base}.csv", index=False)

        # ==================== TAB-EXTREME ====================
        if scope in ('extreme_trend', 'extreme'):
            # ambil timeseries tahunan ekstrem
            descriptor = memoized_process_and_clip_all_scenarios_extreme(aoi, var)
            processed_ext = memoized_analyze_extreme_data(descriptor)
            annual_stats = processed_ext.get("annual_stats_combined", {})
            if not annual_stats:
                return dash.no_update

            df_out = None
            if period in ('historis', 'ssp126', 'ssp245', 'ssp370', 'ssp585'):
                df_out = annual_stats.get(period, pd.DataFrame())
            else:
                # gabungkan semua
                dfs = []
                for scn, df in annual_stats.items():
                    if df is None or df.empty:
                        continue
                    temp = df.copy()
                    temp['scenario'] = scn
                    dfs.append(temp)
                if dfs:
                    df_out = pd.concat(dfs, ignore_index=True)
            if df_out is None or df_out.empty:
                return dash.no_update
            return dcc.send_data_frame(df_out.to_csv, f"{filename_base}.csv", index=False)

        if scope in ('extreme_spatial', 'extreme_map'):
            descriptor = memoized_process_and_clip_all_scenarios_extreme(aoi, var)
            processed_ext = memoized_analyze_extreme_data(descriptor)
            clipped_arrays = descriptor  # sesuai struktur Anda
            scenario = period if period in ('historis','ssp126','ssp245','ssp370','ssp585') else 'historis'
            da = clipped_arrays.get(scenario)
            if da is None:
                return dash.no_update
            try:
                da_mean = da.mean(dim='TIME') if 'TIME' in da.dims else da
            except Exception:
                da_mean = da
            lat_name = next((c for c in da_mean.coords if 'lat' in c.lower()), None)
            lon_name = next((c for c in da_mean.coords if 'lon' in c.lower()), None)
            if lat_name is None or lon_name is None:
                return dash.no_update
            lats = np.asarray(da_mean[lat_name].values)
            lons = np.asarray(da_mean[lon_name].values)
            z = np.squeeze(np.asarray(da_mean.values))
            rows = []
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    rows.append({'lat': float(lat), 'lon': float(lon),
                                 'value': (float(z[i, j]) if not np.isnan(z[i, j]) else None)})
            df_out = pd.DataFrame(rows)
            return dcc.send_data_frame(df_out.to_csv, f"{filename_base}.csv", index=False)

    except Exception as e:
        print("Download error:", e)
        return dash.no_update

    return dash.no_update

@app.callback(
    Output('ocean-popup', 'displayed'),
    Input('main-tabs', 'value')
)
def show_ocean_popup(selected_tab):
    if selected_tab == 'tab-ocean':
        return True   # tampilkan pop up
    return False      # tab lain tidak

if __name__ == "__main__":
    app.run(debug=True)
