import os
import textwrap
from pathlib import Path

import polars as pl
import numpy as np
from datetime import datetime
from dateutil import parser
try:
    from werkzeug.middleware.profiler import ProfilerMiddleware
except:
    print("You are missing the werkzeug package. No issue, unless you want to be profiling.")
# import matplotlib.pyplot as plt

# import re
# import json

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dash_table, dcc, Input, Output, State, \
    callback, callback_context, clientside_callback
from dash.exceptions import PreventUpdate
# import dash_mantine_components as dmc
import dash_bootstrap_components as dbc

pl.enable_string_cache()

# Get the absolute path to the top-level directory
BASE_DIR = Path(__file__).parent.parent

# Path to the assets folder
ASSETS_DIR = BASE_DIR / 'assets'

# Read the Markdown content from the file
with open(f'{ASSETS_DIR}/modal_text.md', 'r', encoding='utf8') as file:
    modal_text = file.read()

regional_office_codes = ['MXR', 'NCR', 'NER', 'SCR', 'SER', 'WXR']
central_office_code = 'BOP'

default_timerange = ['2000-01-01', '2024-06-01'] #datetime.today().strftime('%Y-%m-%d')]

class MetricCard(dbc.Card):
    def __init__(
        self,
        title,
        id,
    ):
        super().__init__(
            children=[
                html.H4("-", id={"type": "metric-value", "index": id}, style={'margin': '0', 'padding': '0'}),
                html.P(title, id={"type": "metric-text", "index": id}, style={'margin': '0', 'padding': '0'}),
            ],
            body=True,
            color='#CED4DA',
            className="my-auto text-center",
        )

complaint_data_dtype_dict = {
    "CASENBR": "int32",
    "ITERLVL": "category",
    "CDFCLEVN": "category",
    "CDFCLRCV": "category",
    "CDOFCRCV": "category",
    "CDSTATUS": "category",
    "STATRSN1": "category",
    "STATRSN2": "category",
    "STATRSN3": "category",
    "STATRSN4": "category",
    "STATRSN5": "category",
    "CDSUB1PR": "category",
    "CDSUB1SC": "category",
    "sdtdue": "datetime64[ns]",
    "sdtstat": "datetime64[ns]",
    "sitdtrcv": "datetime64[ns]",
    "accept": "boolean",
    "reject": "boolean",
    "deny": "boolean",
    "grant": "boolean",
    "other": "boolean",
    "submit": "boolean",
    "filed": "boolean",
    "diffreg_filed": "boolean",
    "diffinst": "boolean",
    "closed": "boolean",
    "comptime": "Int16",
    "timely": "boolean",
    "diffreg_answer": "boolean",
    "overdue": "boolean",
    "untimely": "boolean",
    "resubmit": "boolean",
    "noinfres": "boolean",
    "attachmt": "boolean",
    "wronglvl": "boolean",
    "otherrej": "boolean",
    "cdsub1cb": "category",
}

used_fields = ['ITERLVL','CDFCLRCV','CDOFCRCV','CDSTATUS','sitdtrcv',
               'accept','reject','deny','grant','other','cdsub1cb']

# load the complaint filings data into a polars LazyFrame
cpt_df = pl.scan_parquet(f'{BASE_DIR}/data/complaint-filings-optimized.parquet')#, columns=used_fields)

# cpt_df = pd.read_parquet('https://drive.google.com/uc?export=download&id=1ST06IlcakkLsR-KNoXtop1ut9QbAiDdC',)
# _parquet_kwargs = {"engine": "pyarrow",
#                    "compression": "brotli",
#                    "index": False}
# cpt_df.astype(complaint_data_dtype_dict).to_parquet('../data/complaint-filings-optimized.parquet', **_parquet_kwargs)
# read_mem = cpt_df.memory_usage().sum() / 1024 ** 2
# print(read_mem)

# cpt_df[['sdtdue', 'sdtstat', 'sitdtrcv']] = cpt_df[['sdtdue', 'sdtstat', 'sitdtrcv']].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce',)

name_key_df = pl.read_csv(f'{BASE_DIR}/data/facility-info.csv', schema_overrides={'facility_code': pl.Categorical})

subj_codes_df = pl.read_csv(f'{BASE_DIR}/data/subject-codes-updated.csv')

subj_opts = [
    {'label': row['secondary_desc'], 'category': row['clear_categories'], 'value': row['code']}
    for row in subj_codes_df.iter_rows(named=True)
]
# print(subj_opts)
subj_opts = sorted(subj_opts, key=lambda x: x['label'])
subj_cat_opts_list = sorted(set([subj_dict['category'] for subj_dict in subj_opts]))
subj_cat_redund_list = [subj_dict['category'] for subj_dict in subj_opts]
subj_desc_opts_list = [subj_dict['label'] for subj_dict in subj_opts]
subj_code_opts_list = [subj_dict['value'] for subj_dict in subj_opts]

subj_cats_df = pl.DataFrame({
    'code': pl.Series(subj_code_opts_list, dtype=pl.Categorical),
    'fine_cat': subj_desc_opts_list,
    'gen_cat': subj_cat_redund_list,
})

status_dict = {'CLD': 'Denied', 
               'CLO': 'Closed (Other)', 
               'CLG':'Granted', 
               'ACC':'Accepted', 
               'REJ':'Rejected'}

trp_color = '#1e374f'

color_map_pie = {
    'Rejected':trp_color,
    'Denied':'#DD6E42',
    'Closed (Other)':'#9882AC',
    'Granted':'#FFDEC2'
}
long_color_seq = px.colors.qualitative.Prism + px.colors.qualitative.T10 + px.colors.qualitative.Plotly

color_map_sunburst = {subj_cat:long_color_seq[i] for i, subj_cat in enumerate(subj_cat_opts_list)}

external_stylesheets = [dbc.themes.BOOTSTRAP,  dbc.icons.BOOTSTRAP] # ['https://codepen.io/chriddyp/pen/bWLwgP.css'] #[dbc.themes.BOOTSTRAP,  dbc.icons.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets, assets_folder=str(ASSETS_DIR), title="TRP's Administrative Remedy Dashboard")

server = app.server

app.layout = dbc.Container(
    id='dashboard-container',
    children = [
        dbc.Row([
            dbc.Col([
                dbc.Row(
                    dbc.Col([
                        dbc.Label('Choose filing level:', style={'fontWeight': 'bold'}),
                        dbc.Checklist(
                            id='filing-level',
                            options=[
                                {'label': 'Facility (BP9)', 'value': 'F'},
                                {'label': 'Region (BP10)', 'value': 'R'},
                                {'label': 'Agency (BP11)', 'value': 'A'},
                            ],
                            value=['F', 'R', 'A'],
                            inline=True,
                        ),
                        dcc.Store(data=['F'], id='filing-store')
                    ])
                ),
                dbc.Row(
                    dbc.Col([
                        dbc.Label('Track cases by:', style={'fontWeight': 'bold'}),
                        dbc.Select(
                            id='tracking-level',
                            options=[
                                {'label': 'Institution of Origin', 'value': 'CDFCLRCV'},
                                {'label': 'Office Responsible for Outcome', 'value': 'CDOFCRCV'},
                            ],
                            value='CDFCLRCV',
                        ),
                    ])
                ),
                dbc.Row([
                    dbc.Col(
                        MetricCard("Selected Institution", id="inst-card"),
                        width=8,
                    ),
                    dbc.Col(
                        MetricCard("Cases", id="cases-ticker"),
                        width=4,
                    ),
                ], className='mt-2',),
            ], width=5),
            dbc.Col(
                html.Div(
                    dcc.Graph(
                        id='institution-map',
                        clear_on_unhover=True,
                        style={'height': '100%', 'width': '100%'},
                    ),
                    id='graph-container',
                    style={'height': '100%', 'display': 'flex'},
                ),
                width=7
            ),
        ], className='mt-1'),

        dbc.Row([dbc.Col(html.Hr(), width=12)]),

        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col(
                        dbc.Label('Filter by subject:', style={'fontWeight': 'bold'}),
                        width="auto"
                    ),
                    dbc.Col(
                        html.Div([
                            dbc.DropdownMenu(
                                children=[
                                    dbc.Checklist(
                                        id='subj-cat-filter',
                                        options=subj_cat_opts_list,
                                        value=subj_cat_opts_list,
                                        style={'font-size':'12px', 'overflow-y':'scroll', 'max-height': '100px'},
                                    ),
                                ],
                                color='secondary',
                                direction='down',
                                size='sm',
                                label="SELECT BY CATEGORY",
                                style={'margin-right': '10px'},
                            ),
                            dbc.Button('SELECT ALL', color='secondary', outline=True, id='all-button-subj',
                                       className='all-button', size='sm',
                                       style={'margin-right': '10px', 'font-size': '12px'}),
                            dbc.Button('SELECT NONE', color='secondary', outline=True, id='none-button-subj',
                                       className='none-button', size='sm', style={'font-size': '12px'}),
                        ], style={'display': 'flex', 'justify-content': 'flex-end'}),  # Ensures buttons align right
                        width=True
                    ),
                ], justify="between", align="center"),
                dbc.Row([
                    dbc.Col([
                        dash_table.DataTable(
                            id='datatable-subj-filter',
                            columns=[
                                {'name': '', 'id': 'label'}
                            ],
                            data=subj_opts, #table that I defined at start
                            fixed_rows={'headers': False},
                            filter_action='native',
                            row_selectable="multi",
                            selected_rows=list(range(len(subj_opts))), # not needed done below instead
                            virtualization=False,
                            page_action='none',
                            style_table={
                                'minHeight': '120px',
                                'maxHeight': '200px',
                                'overflowY': 'auto',
                            },
                            css=[
                                {
                                    'selector': '.dash-cell div.dash-cell-value',
                                    'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;',
                                },
                                {
                                    'selector': 'tr:first-child',
                                    'rule':'''
                                            display: None;
                                    '''
                                },
                            ],
                            filter_options={
                                'case': 'insensitive',
                                'placeholder_text': 'Search for specific case subjects...',
                            },
                            style_header={
                                'backgroundColor': trp_color,
                                'color': 'white',
                                'fontSize': '14px',
                                'fontWeight': 'bold',
                                'textAlign': 'center',
                            },
                            style_cell={
                                'whiteSpace': 'no-wrap',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                                'maxWidth': 0,
                                'fontSize': '12px',
                                'textAlign': 'left',
                            },
                            tooltip_data=[
                                {
                                    column: {'value': str(value), 'type': 'markdown'}
                                    for column, value in row.items()
                                } for row in subj_opts
                            ],
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(232, 232, 232)'
                                }
                            ],
                            style_as_list_view=True,
                        ),
                    ]),
                ]),
            ], width=4),
            dbc.Col(
                dcc.Graph(
                    id='institution-sunburst',
                ),
                width=4,
            ),
            dbc.Col(
                dcc.Graph(
                    id='institution-pie',
                    figure={'layout': go.Layout(margin=dict(l=10, r=10, t=10, b=10))}
                ),
                width=4
            )
        ], className='my-auto'),

        dbc.Row([
            dbc.Col(
                dcc.Graph(
                    id='case-cts',
                    figure={
                        'layout': go.Layout(
                            margin=dict(l=0, r=0, t=0, b=0),  # Tight margins
                        )
                    }
                ),
                width=12,
            )
        ]),
        html.Div(
            dbc.Button(
                html.I(className="bi bi-info-circle"),
                id="open-modal-button",
                color="rgb(232, 232, 232)",
                style={
                    "borderRadius": "50%",  # Make it a circle
                    "width": "50px",        # Ensure equal width and height
                    "height": "50px",
                    "display": "flex",      # Center the icon
                    "justifyContent": "center",
                    "alignItems": "center",
                    "padding": "0",         # Remove extra padding
                    "font-size": "40px",
                    "backgroundColor": "transparent",
                },
            ),
            style={
                "position": "fixed",
                "bottom": "20px",
                "right": "20px",
                "zIndex": 1049, # modal zindex default is 1050
            },
        ),

        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Info")),
            dbc.ModalBody(dcc.Markdown(modal_text)),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-modal-button", className="ms-auto", color='secondary', outline=True))
        ], id="help-modal", size='lg', is_open=True),

        dcc.Store(id='time_range', data=default_timerange),
    ],
    fluid=True,
)

# Callbacks to manage modal behavior
@app.callback(
    Output("help-modal", "is_open"),
    [Input("open-modal-button", "n_clicks"), Input("close-modal-button", "n_clicks")],
    [State("help-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(open_click, close_click, is_open):
    if open_click or close_click:
        return not is_open
    return is_open

@app.callback(
    Output({'type': 'metric-value', 'index': 'inst-card'}, 'children'),
    inputs=[
        Input('institution-map', 'hoverData'),
        Input('institution-map', 'clickData'),
    ],
)
def update_inst_card(hoverData, clickData,):
    # If hoverData changes, only update pie if there is no clickData
    if ((clickData is not None) and
            ('institution-map.hoverData' in callback_context.triggered_prop_ids) and
            (len(callback_context.triggered_prop_ids) <= 1)):
        raise PreventUpdate()

    info = clickData if clickData else hoverData  # hoverData if hoverData else clickData

    if info is None:
        inst_name = 'All Institutions'
    else:
        inst_code = info['points'][0]['customdata'][3]
        inst_name = name_key_df.filter(pl.col('facility_code') == inst_code)['nice_name'][0]

    return inst_name

@app.callback(
    output = Output('time_range', 'data'),
    inputs = [
        Input('case-cts', 'relayoutData'),
    ],
    state = [
        State('time_range', 'data'),
    ],
)
def update_time_range(casects_relayout, time_range):
    if casects_relayout:
        time_range = casects_relayout.get(
            'xaxis.range',
            [
                casects_relayout.get('xaxis.range[0]', default_timerange[0]),
                casects_relayout.get('xaxis.range[1]', default_timerange[1])
            ]
        )

    return time_range

@app.callback(
    [
        Output('datatable-subj-filter', 'selected_rows', allow_duplicate=True,),
        Output('subj-cat-filter', 'value', allow_duplicate=True,),
    ],
    [
        Input('datatable-subj-filter', 'selected_rows'),
        Input('subj-cat-filter', 'value'),
    ],
    prevent_initial_call=True,
)
def update_subj_filter_by_category(selected_rows, selected_cats):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'subj-cat-filter':
        new_selected_rows = []
        for i in range(len(subj_opts)):
            for cat in selected_cats:
                if subj_opts[i]['category'] == cat:
                    new_selected_rows.append(i)
                    continue
        return new_selected_rows, selected_cats
    # if changing the fine category
    elif trigger_id == 'datatable-subj-filter':
        if len(selected_rows) == len(subj_opts):
            return selected_rows, subj_cat_opts_list
        # unselect and select general categories based on if all fine elements of the gen_cat are selected
        else:
            selected_subjs = [subj_opts[subj_ind]['label'] for subj_ind in selected_rows]
            possible_cats = list(set(subj_cats_df.filter(pl.col('fine_cat').is_in(selected_subjs))['gen_cat']))
            for possible_cat in possible_cats:
                subjs_in_cat = list(subj_cats_df.filter(subj_cats_df['gen_cat']==possible_cat)['fine_cat'])
                print(subjs_in_cat)
                selected_subj_in_cat_count = 0
                for selected_subj in selected_subjs:
                    if selected_subj in subjs_in_cat:
                        selected_subj_in_cat_count += 1
                        print(selected_subj_in_cat_count)
                if selected_subj_in_cat_count != len(subjs_in_cat):
                    print(f'removing {possible_cat}')
                    possible_cats.remove(possible_cat)
                    print(possible_cats)
            return selected_rows, possible_cats

@app.callback(
    [
        Output('datatable-subj-filter', 'selected_rows'),
        Output('subj-cat-filter', 'value'),
    ],
    [
        Input('all-button-subj', 'n_clicks'),
        Input('none-button-subj', 'n_clicks'),
    ],
    prevent_initial_call=True,
)
def select_all_subj(all_clicks, none_clicks):
    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'all-button-subj':
        return [i for i in range(len(subj_opts))], subj_cat_opts_list
    else:
        return [], []

@app.callback(
    Output('institution-map', 'clickData'),
    [Input('graph-container', 'n_clicks')]
)
def reset_clickData(n_clicks):
    return None

@app.callback(
    Output('filing-level', 'value'),
    Output('filing-store', 'data'),
    Input('filing-level', 'value'),
    State('filing-store', 'data')
)
def update_checklist(value, active):
    """
    Prevent checklist from having no filing levels selected.
    """
    if len(value) < 1:
        return active, active
    else:
        return value, value

@app.callback(
    Output('institution-map', 'figure'),
    inputs = [
        Input('filing-level', 'value'),
        Input('tracking-level', 'value'),
        Input('datatable-subj-filter', "selected_rows"),
        Input('time_range', 'data'),
    ],
    state = [
        State('datatable-subj-filter', "data"),
    ],
)
def update_map(filingSelections, trackingSelection, selected_subj_rows, time_range, subj_rows):
    selected_subj_list = [subj_rows[i]['value'] for i in selected_subj_rows]
    time_start_dt = parser.parse(time_range[0])
    time_end_dt = parser.parse(time_range[1])
    time_start_str = time_start_dt.strftime('%m/%Y')
    time_end_str = time_end_dt.strftime('%m/%Y')

    filter_expr = (
            (pl.col('ITERLVL').is_in(filingSelections)) &
            (pl.col('cdsub1cb').is_in(selected_subj_list)) &
            (pl.col('sitdtrcv').is_between(time_start_dt, time_end_dt))
    )

    filtered_count = (
        cpt_df
        .filter(
            filter_expr
        )
        .select(pl.len())  # Count rows after filter
        .collect()  # Collect the count result immediately
    ).item()  # Extract the integer value from the result

    summary_df = (
        cpt_df
        .filter(
            filter_expr
        )
        .group_by(trackingSelection)
        .agg([
            pl.col('CDSTATUS').len().alias('total_cases'),
            pl.col('reject').sum().alias('rejected_cases'),
            pl.col('deny').sum().alias('denied_cases'),
            pl.col('grant').sum().alias('granted_cases'),
            pl.col('other').sum().alias('closed_other_cases'),
            pl.col('accept').sum().alias('accepted_cases')
        ])
        .with_columns([
            (pl.col('rejected_cases') + pl.col('denied_cases') + pl.col('granted_cases') + pl.col('closed_other_cases'))
            .alias('total_closed_cases'),
            (1 - (pl.col('granted_cases') /
                  (pl.col('rejected_cases') + pl.col('denied_cases') + pl.col('granted_cases') + pl.col(
                      'closed_other_cases'))))
            .alias('no_remedy_frac')
        ])
    )

    summary_df = summary_df.collect()

    # Join with `name_key_df`
    summary_df = summary_df.join(name_key_df, left_on=trackingSelection, right_on='facility_code', coalesce=False)

    # Filter rows where 'latitude' is not null
    summary_df = summary_df.filter(pl.col('latitude').is_not_null())

    # Add hover_template column
    summary_df = summary_df.with_columns(
        pl.when(pl.col('pop_total').is_not_null())
        .then(
            pl.lit(
                "<b>%{hovertext}</b><br>" +
                f"2024 Population: " + "%{customdata[0]:,}<br>" +
                f"Total cases ({time_start_str}-{time_end_str}): " + "%{customdata[1]:,}<br>" +
                "Non-approval Rate: %{customdata[2]:.1%}<br>" +
                "<extra></extra>"
            )
        )
        .otherwise(
            pl.lit(
            "<b>%{hovertext}</b><br>" +
                f"Total cases ({time_start_str}-{time_end_str}): " + "%{customdata[1]:,}<br>" +
                "Non-approval Rate: %{customdata[2]:.1%}<br>" +
                "<extra></extra>"
            )
        ).alias('hover_template')
    )

    # Split data based on category
    region_mask = summary_df[trackingSelection].is_in(regional_office_codes)
    central_mask = summary_df[trackingSelection].is_in([central_office_code])

    dff_F = summary_df.filter(~(region_mask | central_mask))
    dff_R = summary_df.filter(region_mask)
    dff_A = summary_df.filter(central_mask)

    sizemax = 20
    casetotalmax = filtered_count/50 #np.sum(filter_mask)/50 #np.max(summary_df['total_closed_cases'])/10

    # Create the mapbox figure with multiple traces
    fig = go.Figure()

    for test_df, test_cscale, locality in zip([dff_F, dff_R, dff_A], ['Reds', 'Greens', 'Blues'], ['Facility', 'Regional Office', 'BOP Headquarters']):
        fig.add_trace(go.Scattermap(
            lat=test_df['lat_adj'],
            lon=test_df['long_adj'],
            mode='markers',
            marker=go.scattermap.Marker(
                size=test_df['total_closed_cases'],
                color=test_df['no_remedy_frac'],
                colorscale=test_cscale, # Use Reds or another color scale if necessary
                cmin=0.5,
                cmax=1.0,
                sizeref=(2 * casetotalmax)/(sizemax**2),
                sizemode='area',
                sizemin=2,
            ),
            name=locality,
            hoverinfo='text',
            hovertext=test_df['nice_name'],
            customdata=test_df[['pop_total', 'total_closed_cases', 'no_remedy_frac', 'facility_code']],
            hovertemplate=test_df['hover_template'],
        ))
    
    # Update layout
    fig.update_layout(
        map_style="basic",
        map_zoom=2.7,
        map_center={"lat": 38, "lon": -95},
        margin={"t":0,"b":0,"r":0,"l":0},
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        uirevision='constant',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0)',
        ),
        # height=300,
    )
    
    return fig
    

@app.callback(
    Output('institution-pie', 'figure'),
    Output({'type': 'metric-value', 'index': 'cases-ticker'}, 'children'),
    inputs = [
        Input('institution-map', 'hoverData'),
        Input('institution-map', 'clickData'),
        Input('filing-level', 'value'),
        Input('tracking-level', 'value'),
        Input('datatable-subj-filter', "selected_rows"),
        Input('time_range', 'data'),
    ],
    state = [
        State('datatable-subj-filter', "data"),
    ],
)
def update_pie(hoverData,clickData,filingSelections,trackingSelection,selected_subj_rows, time_range, subj_rows):
    # If hoverData changes, only update pie if there is no clickData
    if ((clickData is not None) and
        ('institution-map.hoverData' in callback_context.triggered_prop_ids) and
        (len(callback_context.triggered_prop_ids) <= 1)):
        raise PreventUpdate()

    info = clickData if clickData else hoverData #hoverData if hoverData else clickData
    selected_subj_list = [subj_rows[i]['value'] for i in selected_subj_rows]

    time_start_dt = parser.parse(time_range[0])
    time_end_dt = parser.parse(time_range[1])

    filter_expr = (
            (pl.col('ITERLVL').is_in(filingSelections)) &
            (pl.col('cdsub1cb').is_in(selected_subj_list)) &
            (pl.col('sitdtrcv').is_between(time_start_dt, time_end_dt))
    )

    if info is not None:
        inst_code = info['points'][0]['customdata'][3]
        filter_expr &= (pl.col(trackingSelection) == inst_code)

    counts_df = (
        cpt_df
        .filter(filter_expr)
        .group_by('CDSTATUS')
        .agg(pl.len().alias('values')) #.len().alias('value')
        .filter(~pl.col('CDSTATUS').eq('ACC'))  # Exclude 'ACC' status if it exists
        .sort(pl.col('CDSTATUS').cast(pl.Enum(['CLG','CLO','CLD','REJ'])))
    )
    counts_df = counts_df.collect()

    labels = [status_dict[status] for status in counts_df['CDSTATUS']]
    
    colors = [color_map_pie.get(label, 'gray') for label in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=counts_df['values'],
                # automargin=False,
            )
        ],
    )
    fig.update_traces(
        # hoverinfo='label+percent',
        hovertemplate="<b>%{label}</b><br>Cases: %{value:,}</br>%{percent}",
        name='', #gets rid of secondary box
        textinfo='label',
        textposition='inside',
        # text=[val for val in counts_df.values],
        # textfont_size=18,
        pull=[0.3,0,0,0] if 'CLG' in counts_df['CDSTATUS'] else [0,0,0,0],
        sort=False, rotation=270,
        marker=dict(colors=colors, line=dict(color='#000000', width=1))
    )
    fig.update_layout(
        title='Administrative Remedy Outcomes',
        showlegend=False,
        margin={"t": 30, "b": 0, "l": 0, "r": 0},
        # height=300,
    )
                    
    return fig, f'{counts_df["values"].sum():,}'

@app.callback(
    Output('institution-sunburst', 'figure'),
    inputs = [
        Input('institution-map', 'hoverData'),
        Input('institution-map', 'clickData'),
        Input('filing-level', 'value'),
        Input('tracking-level', 'value'),
        Input('datatable-subj-filter', "selected_rows"),
        Input('time_range', 'data'),
    ],
    state = [
        State('datatable-subj-filter', "data"),
    ],
)
def update_sunburst(hoverData,clickData,filingSelections,trackingSelection,selected_subj_rows, time_range, subj_rows):
    # If hoverData changes, only update pie if there is no clickData
    if ((clickData is not None) and
        ('institution-map.hoverData' in callback_context.triggered_prop_ids) and
        (len(callback_context.triggered_prop_ids) <= 1)):
        raise PreventUpdate()

    info = clickData if clickData else hoverData #hoverData if hoverData else clickData
    selected_subj_list = [subj_rows[i]['value'] for i in selected_subj_rows]

    time_start_dt = parser.parse(time_range[0])
    time_end_dt = parser.parse(time_range[1])
    
    filter_expr = (
            (pl.col('ITERLVL').is_in(filingSelections)) &
            (pl.col('cdsub1cb').is_in(selected_subj_list)) &
            (pl.col('sitdtrcv').is_between(time_start_dt, time_end_dt))
    )

    if info is not None:
        inst_code = info['points'][0]['customdata'][3]
        filter_expr &= (pl.col(trackingSelection) == inst_code)

    subj_cts_df = (
        cpt_df
        .filter(filter_expr)
        .group_by('cdsub1cb')
        .agg(pl.len().alias('values'))
    )
    subj_cts_df = subj_cts_df.collect()

    subj_cts_df = subj_cts_df.join(subj_cats_df, left_on='cdsub1cb', right_on='code', coalesce=True)

    gen_cat_cts_df = subj_cts_df.group_by('gen_cat').agg(pl.col('values').sum()).sort(pl.col('gen_cat'))

    labels = gen_cat_cts_df['gen_cat'].to_list() + subj_cts_df['fine_cat'].to_list()
    parents = [''] * len(gen_cat_cts_df) + subj_cts_df['gen_cat'].to_list()
    values = gen_cat_cts_df['values'].to_list() + subj_cts_df['values'].to_list()

    color_map_sequence = [color_map_sunburst[cat] for cat in gen_cat_cts_df['gen_cat'].to_list()]

    # Format hover text with text wrapping
    def format_hover_text(label, value):
        wrapped_label = '<br>'.join(textwrap.wrap(label, width=30))
        return f"{wrapped_label}<br><b>Cases: {value:,}</b>"

    hover_texts = [format_hover_text(label, val) for label, val in zip(labels, values)]

    # Create Sunburst Plot
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",  # Ensures values sum up properly
        marker=dict(colors=color_map_sequence),
        hovertext=hover_texts,
        hoverinfo='text',
        insidetextorientation='radial',
    ))

    # Update Layout
    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        # uniformtext=dict(minsize=6, mode='hide'),
    )


    return fig


@app.callback(
    Output('case-cts', 'figure'),
    inputs = [
        Input('institution-map', 'hoverData'),
        Input('institution-map', 'clickData'),
        Input('filing-level', 'value'),
        Input('tracking-level', 'value'),
        Input('datatable-subj-filter', "selected_rows"),
    ],
    state = [
        State('datatable-subj-filter', "data"),
        State('time_range', 'data'),
    ],
)
def update_case_counts(hoverData, clickData, filingSelections, trackingSelection,selected_subj_rows, subj_rows, time_range):
    # If hoverData changes, only update pie if there is no clickData
    if ((clickData is not None) and
            ('institution-map.hoverData' in callback_context.triggered_prop_ids) and
            (len(callback_context.triggered_prop_ids) <= 1)):
        raise PreventUpdate()

    info = clickData if clickData else hoverData  # hoverData if hoverData else clickData
    selected_subj_list = [subj_rows[i]['value'] for i in selected_subj_rows]

    # filter_mask = cpt_df['ITERLVL'].isin(filingSelections)
    # filter_mask &= cpt_df['cdsub1cb'].isin(selected_subj_list)
    filter_expr = (
            (pl.col('ITERLVL').is_in(filingSelections)) &
            (pl.col('cdsub1cb').is_in(selected_subj_list))
    )

    if info is not None:
        inst_code = info['points'][0]['customdata'][3]
        filter_expr &= (pl.col(trackingSelection) == inst_code)

    case_counts_df = (
        cpt_df
        .filter(filter_expr)
        .sort('sitdtrcv')
        .group_by_dynamic('sitdtrcv', every='1w',start_by='datapoint')  # Weekly resampling
        .agg(pl.len().alias('case_count'))
        .select(['sitdtrcv', 'case_count'])  # Keep only necessary columns
        .collect()
    )

    # can only upsample if DataFrame isn't empty
    if len(case_counts_df) > 0:
        # filling in gaps and convert to pandas
        #### see: https://www.rhosignal.com/posts/filling-gaps-lazy-mode/
        case_counts_df = (
            case_counts_df
            .upsample('sitdtrcv', every='1w')
            .fill_null(strategy='zero')
        )

    case_counts_df = case_counts_df.with_columns([
        pl.col("case_count").rolling_mean(window_size=4).alias("monthly_rolling_avg"),
        pl.col("case_count").rolling_sum(window_size=4, min_periods=1).alias("monthly_rolling_sum")
    ])


    fig = go.Figure()

    # Add the actual event counts to the plot
    # fig.add_trace(go.Scatter(x=case_counts_df['sitdtrcv'], y=case_counts_df['case_count'],
    #                          mode='lines', name='Filing Count', line=dict(color=color_map_pie.get('Rejected'))))

    # Add the rolling average line to the plot
    fig.add_trace(go.Scatter(x=case_counts_df['sitdtrcv'], y=case_counts_df['monthly_rolling_sum'],
                             mode='lines', name='2-Month Rolling Average',
                             line=dict(color=trp_color, width=2)))

    fig.update_layout(
        title=f"Rolling Monthly Administrative Remedy Filings",
        # xaxis_title="Time",
        # yaxis_title="Weekly Filing Count",
        xaxis = dict(
            rangeslider = {'visible': True,},# 'range':default_timerange},
            range = time_range if time_range!=default_timerange else default_timerange, # keep time_range permanently to keep everything 2000-2024
            autorange = False, #if time_range!=default_timerange else True, # keep false permanently to keep everything 2000-2024
            # autorangeoptions = {'minallowed':default_timerange[0], 'maxallowed':default_timerange[1]},
        ),
        margin={"t": 40, "b": 0, "l": 0, "r": 5},
        height=250,
    )

    return fig
    
if __name__ == "__main__":

    # see https://community.plotly.com/t/performance-profiling-dash-apps-with-werkzeug/65199
    if os.getenv("PROFILER", None):
        app.server.config["PROFILE"] = True
        app.server.wsgi_app = ProfilerMiddleware(
            app.server.wsgi_app,
            sort_by=("tottime", "cumtime"),
            restrictions=[50],
            stream=None,
            profile_dir='./profiling',
        )

    app.run(debug=True, port=8051)
