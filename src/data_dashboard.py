import os

import polars as pl
import numpy as np
from datetime import datetime
try:
    from werkzeug.middleware.profiler import ProfilerMiddleware
except:
    print("You are missing the werkzeug package. No issue, unless you want to be profiling.")
# import matplotlib.pyplot as plt

# import re
# import json

# import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dash_table, dcc, Input, Output, State, \
    callback, callback_context
from dash.exceptions import PreventUpdate

pl.enable_string_cache()

regional_office_codes = ['MXR', 'NCR', 'NER', 'SCR', 'SER', 'WXR']
central_office_code = 'BOP'

default_timerange = ['2000-01-01', '2024-06-01'] #datetime.today().strftime('%Y-%m-%d')]

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
cpt_df = pl.scan_parquet('../data/complaint-filings-optimized.parquet')#, columns=used_fields)

# cpt_df = pd.read_parquet('https://drive.google.com/uc?export=download&id=1ST06IlcakkLsR-KNoXtop1ut9QbAiDdC',)
# _parquet_kwargs = {"engine": "pyarrow",
#                    "compression": "brotli",
#                    "index": False}
# cpt_df.astype(complaint_data_dtype_dict).to_parquet('../data/complaint-filings-optimized.parquet', **_parquet_kwargs)
# read_mem = cpt_df.memory_usage().sum() / 1024 ** 2
# print(read_mem)

# cpt_df[['sdtdue', 'sdtstat', 'sitdtrcv']] = cpt_df[['sdtdue', 'sdtstat', 'sitdtrcv']].apply(pd.to_datetime, format='%Y-%m-%d', errors='coerce',)

name_key_df = pl.read_csv('../data/facility-info.csv', schema_overrides={'facility_code': pl.Categorical})

subj_codes_df = pl.read_csv('https://drive.google.com/uc?export=download&id=1OQ8xLLF3hG3Dtd9C_LJpvngfsH3YO-5B')

subj_code_opts = [
    {'label': row['secondary_desc'], 'value': row['code']}
    for row in subj_codes_df.iter_rows(named=True)
]
subj_code_opts = sorted(subj_code_opts, key=lambda x: x['label'])
# subj_code_opts = [{'label': 'SELECT ALL', 'value': 'all'}] + subj_code_opts

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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Span('Choose filing level:',
                          style={'fontWeight': 'bold', 'margin-right': '20px'}),
                dcc.Checklist(
                    id='filing-level',
                    options=[
                        {
                            'label': 'Facility (BP9)',
                            'value': 'F'
                        },
                        {
                            'label': 'Region (BP10)',
                            'value': 'R'
                        },
                        {
                            'label': 'Agency (BP11)',
                            'value': 'A'
                        },
                    ],
                    value=['F', 'R', 'A'],
                    inline=True,
                ),
                dcc.Store(data=['F'], id='filing-store')
            ],
            style={'width': '100%', 'margin-bottom': '40px'}),

            html.Div([
                html.Span('Track cases by:',
                          style={'fontWeight': 'bold', 'margin-right': '20px'}),
                dcc.Dropdown(
                    id='tracking-level',
                    options=[
                        {
                            'label': 'Institution of Origin',
                            'value': 'CDFCLRCV'
                        },
                        {
                            'label': 'Office Responsible for Outcome',
                            'value': 'CDOFCRCV'
                        },
                    ],
                    clearable=False,
                    value='CDOFCRCV' #'CDFCLRCV',#'CDOFCRCV',
                ),
            ],
            style={'width': '100%'}),
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top', 'padding-right': '20px'}),
        # html.Div(
        #     [
        #         'Filter by case subject:',
        #         dcc.Dropdown(
        #             id="type-dropdown",
        #             optionHeight=55,
        #             options=subj_code_opts,
        #             value=['all'],
        #             multi=True,
        #         ),
        #     ],
        #     style={'width': '33%', 'display': 'inline-block', 'vertical-align':'top'},
        # ),
        html.Div([
            # html.Div([
            #     html.Span('Filter cases by subject:', style={'fontWeight': 'bold', 'margin-right': '20px'}),
            #     html.Button('Select All', id='all-button-genre', className='all-button' ),
            #     html.Button('Select None', id='none-button-genre', className='none-button'),
            # ],
            #     className="multi-filter",
            #     style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'},
            # ),
            html.Div([
                # Left-aligned text
                html.Span('Filter cases by subject:',
                          style={'fontWeight': 'bold', 'margin-right': '20px'}),  # Label

                # Buttons pushed to the right
                html.Div([
                    html.Button('Select All', id='all-button-subj', className='all-button',
                                style={'margin-right': '10px', 'padding': '0 10px', 'height': '25px',
                                       'font-size':'12px','line-height':'8px', 'vertical-align':'middle'}),
                    html.Button('Select None', id='none-button-subj', className='none-button',
                                style={'padding': '0 10px', 'height': '25px',
                                       'font-size':'12px','line-height':'8px', 'vertical-align':'middle'}),
                ], style={'display': 'flex'}),  # Inline-flex for the buttons
            ], style={
                'display': 'flex',
                'justify-content': 'space-between',  # Push the buttons to the right
                'align-items': 'bottom',  # Vertically align both the text and buttons
                'margin-bottom': '10px'
            }),

            html.Div([
                dash_table.DataTable(
                    id='datatable-subj-filter',
                    columns=[
                        {"name": '', "id": 'label'}
                    ],
                    data=subj_code_opts, #table that I defined at start
                    fixed_rows={'headers': False},
                    filter_action='native',
                    row_selectable="multi",
                    selected_rows=list(range(len(subj_code_opts))), # not needed done below instead
                    virtualization=False,
                    page_action='none',
                    style_table={
                        'minHeight': '120px',
                        'maxHeight': '120px',
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
                        'textAlign': 'left',
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(232, 232, 232)'
                        }
                    ],
                    style_as_list_view=True,
                ),
                # html.Div(id='datatable-interactivity-container')
            ],
            ),
        ],
        className='individual-filter',
        style={'width': '50%', 'display': 'inline-block', 'vertical-align':'top'},
        ),
    ], style={'display': 'flex', 'width': '100%'}),

    html.Div([
        html.Hr(
            style={'width': '100%', 'padding':'0px',},
        )
    ], style={'width': '100%', 'padding':'0px',}),
    # html.Div([
    #     dcc.Graph(id='institution-map')
    # ], style={'width': '75%', 'display': 'inline-block'}),
    html.Div([
        html.Div(
            html.Div([
                dcc.Graph(id='institution-map', clear_on_unhover=True)
            ]),
            id='graph-container',
            style={'width': '50%', 'display': 'inline-block',  'padding':'0px',}
        ),

        html.Div(
            dcc.Graph(
                id='institution-pie',
                figure={
                    'layout': go.Layout(
                        margin=dict(l=10, r=10, t=10, b=10),  # Tight margins
                    )
                }
            ),
            style={'width': '50%', 'display': 'inline-block', 'padding':'0px'}),
        ], style={'height':'300px'},
    ),
    html.Div(
        dcc.Graph(
            id='case-cts',
            figure={
                'layout': go.Layout(
                    margin=dict(l=0, r=0, t=0, b=0),  # Tight margins
                )
            }
        ),
        style={'width': '100%', 'display': 'inline-block', 'padding':'0px'}),

    # html.Div(dcc.RangeSlider(
    #     cpt_df['sitdtrcv'].min(),
    #     cpt_df['sitdtrcv'].min(),
    #     step=1,
    #     id='crossfilter-year--slider',
    #     value=df['Year'].max(),
    #     marks={str(year): str(year) for year in df['Year'].unique()}
    # ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
    dcc.Store(id='time_range', data=default_timerange),
])

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
        Output('datatable-subj-filter', "selected_rows"),
    ],
    [
        Input('all-button-subj', 'n_clicks'),
        Input('none-button-subj', 'n_clicks'),
    ],
    [
        State('datatable-subj-filter', "data"), #"derived_virtual_data"),
    ]
)
def select_all_subj(all_clicks, none_clicks, selected_rows):
    if selected_rows is None:
        return [[]]
    ctx = callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
        return [list(range(len(subj_code_opts)))]
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'all-button-subj':
        return [[i for i in range(len(selected_rows))]]
    else:
        return [[]]

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
    selected_subj_code_list = [subj_rows[i]['value'] for i in selected_subj_rows]
    time_start_dt = datetime.strptime(time_range[0],
                                      '%Y-%m-%d %H:%M:%S.%f' if len(time_range[0].split(' ')) > 1 else '%Y-%m-%d')
    time_end_dt = datetime.strptime(time_range[1],
                                    '%Y-%m-%d %H:%M:%S.%f' if len(time_range[1].split(' ')) > 1 else '%Y-%m-%d')
    time_start_str = time_start_dt.strftime('%m/%Y')
    time_end_str = time_end_dt.strftime('%m/%Y')
    # filter_mask = cpt_df['ITERLVL'].isin(filingSelections) & \
    #               cpt_df['cdsub1cb'].isin(selected_subj_code_list) & \
    #               (cpt_df['sitdtrcv'] > time_range[0]) & (cpt_df['sitdtrcv'] < time_range[1])

    filter_expr = (
            (pl.col('ITERLVL').is_in(filingSelections)) &
            (pl.col('cdsub1cb').is_in(selected_subj_code_list)) &
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
                "Rejection/Denial Rate: %{customdata[2]:.1%}<br>" +
                "<extra></extra>"
            )
        )
        .otherwise(
            pl.lit(
            "<b>%{hovertext}</b><br>" +
                f"Total cases ({time_start_str}-{time_end_str}): " + "%{customdata[1]:,}<br>" +
                "Rejection/Denial Rate: %{customdata[2]:.1%}<br>" +
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
    
    # # Trace for facility (Reds colorscale)
    # fig.add_trace(go.scattermap(
    #     lat=dff_F['lat_adj'],
    #     lon=dff_F['long_adj'],
    #     mode='markers',
    #     marker=go.scattermapbox.Marker(
    #         size=dff_F['total_closed_cases'],
    #         color=dff_F['no_remedy_frac'],
    #         colorscale='Greens',
    #         cmin=0.9,  # Set min value for color scaling
    #         cmax=1.0,  # Set max value for color scaling
    #         showscale=False,  # Show the colorscale for this trace
    #         # colorbar=dict(title='Greens', x=0.85)  # Position of colorbar
    #         # sizemax=20,
    #         sizeref=20/np.max(dff_F['total_closed_cases']),
            
    #     ),
    #     text=dff_F["facility_name"], 
    #     customdata=[dff_F['pop_total'], dff_F['total_closed_cases'], dff_F['no_remedy_frac'], dff_F['facility_code']],
    #     name='Facility'
    # ))
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
    # fig.add_trace(go.Scattermap(
    #     lat=dff_F['lat_adj'],
    #     lon=dff_F['long_adj'],
    #     mode='markers',
    #     marker=go.scattermap.Marker(
    #         size=dff_F['total_closed_cases'],
    #         color=dff_F['no_remedy_frac'],
    #         colorscale='Reds', # Use Reds or another color scale if necessary
    #         cmin=0.9,
    #         cmax=1.0,
    #         sizeref=(2 * np.max(dff_F['total_closed_cases']))/(sizemax**2),
    #         sizemode='area',
    #         # sizemin=4,
    #     ),
    #     hoverinfo='text',
    #     hovertext=dff_F['nice_name'],
    #     customdata=dff_F[['pop_total', 'total_closed_cases', 'no_remedy_frac', 'facility_code']],
    #     hovertemplate=dff_F['hover_template'],
    # ))
    
    # Update layout with Mapbox style and settings
    fig.update_layout(
        map_style="basic",
        map_zoom=2.7,
        map_center={"lat": 38, "lon": -95},
        margin={"t":0,"b":0,"r":0,"l":0},
    )

    
    # fig = px.scatter_map(summary_df,
    #                      lat="lat_adj", lon="long_adj", size="total_closed_cases", color="no_remedy_frac", 
    #                      color_continuous_scale=summary_df['color_scale'],#'Reds', 
    #                      size_max=20, hover_name="facility_name", zoom=2.7, range_color=[0.9,1.0],
    #                      hover_data=['pop_total', 'total_closed_cases', 'no_remedy_frac', 'facility_code'],
    #                     )
    # fig.update_layout(
    #     coloraxis={
    #         'colorbar': {
    #             'title': 'Rejection / Denial<br>Rate',
    #             'tickformat':'.0%',
    #             # 'cmin': 0.9,   # Set minimum value
    #             # 'cmax': 1.0  # Set maximum value
    #         }
    #     },
    #     margin=dict(l=0, r=0, t=0, b=0)
    # )
    # fig.update_layout(coloraxis_showscale=False)
    # fig.update_traces(
    #     hovertemplate=summary_df['hover_template'],
    #     # "<b>%{hovertext}</b><br>" + 
    #     #               "2024 Population: %{customdata[0]}<br>" +
    #     #               "Total cases (2000-2007): %{customdata[1]}<br>" +
    #     #               "Rejection/Denial Rate: %{customdata[2]:.1%}<br>" +
    #     #               "<extra></extra>", 
    #     # customdata=name_key_df[['pop_total']],  # Pass customdata for hovertemplate to access
    #     # custom_data=[summary_df['pop_total'], summary_df['total_closed_cases'], summary_df['no_remedy_frac']],  # Pass customdata for hovertemplate to access
    #     hovertext=summary_df['nice_name']  # Preserve the facility name in bold
    # )
    
    fig.update_layout(
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
        height=300,
    )
    
    return fig
    

@app.callback(
    Output('institution-pie', 'figure'),
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
    # if casects_relayout:
    #     time_range = casects_relayout.get('xaxis.range', default_timerange)
    # else:
    #     time_range = default_timerange

    # If hoverData changes, only update pie if there is no clickData
    if ((clickData is not None) and
        ('institution-map.hoverData' in callback_context.triggered_prop_ids) and
        (len(callback_context.triggered_prop_ids) <= 1)):
        raise PreventUpdate()

    info = clickData if clickData else hoverData #hoverData if hoverData else clickData
    selected_subj_code_list = [subj_rows[i]['value'] for i in selected_subj_rows]

    time_start_dt = datetime.strptime(time_range[0],
                                      '%Y-%m-%d %H:%M:%S.%f' if len(time_range[0].split(' ')) > 1 else '%Y-%m-%d')
    time_end_dt = datetime.strptime(time_range[1],
                                    '%Y-%m-%d %H:%M:%S.%f' if len(time_range[1].split(' ')) > 1 else '%Y-%m-%d')

    filter_expr = (
            (pl.col('ITERLVL').is_in(filingSelections)) &
            (pl.col('cdsub1cb').is_in(selected_subj_code_list)) &
            (pl.col('sitdtrcv').is_between(time_start_dt, time_end_dt))
    )
    # filter_mask = cpt_df['ITERLVL'].isin(filingSelections)
    # filter_mask &= cpt_df['cdsub1cb'].isin(selected_subj_code_list)
    # filter_mask &= (cpt_df['sitdtrcv'] > time_range[0]) & (cpt_df['sitdtrcv'] < time_range[1])
    
    # dff = cpt_df[cpt_df['ITERLVL'].isin(filingSelections)]
    if info is None:
        inst_name = 'All Institutions'
    else:
        inst_code = info['points'][0]['customdata'][3]
        # filter_mask &= (cpt_df[trackingSelection] == inst_code)
        filter_expr &= (pl.col(trackingSelection) == inst_code)
        inst_name = name_key_df.filter(pl.col('facility_code') == inst_code)['nice_name'][0]

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

    # fig = px.pie(counts_df, 
    #              values = counts_df.values,
    #              names=[status_dict[status] for status in counts_df.index], 
    #              title='Case Results',
    #              color=[status_dict[status] for status in counts_df.index],
    #              color_discrete_map={'Rejected':'#1e374f',
    #                              'Denied':'#DD6E42',
    #                              'Closed (Other)':'#9882AC',
    #                              'Granted':'#FFE8D4'},
    #              sort=False,
    #             )
    fig = go.Figure(data=[go.Pie(labels=labels,
                                 values=counts_df['values'])])
    fig.update_traces(hoverinfo='label+percent', 
                      textinfo='value', 
                      # text=[val for val in counts_df.values],
                      textfont_size=18,
                      pull=[0.3,0,0,0] if 'CLG' in counts_df['CDSTATUS'] else [0,0,0,0],
                      sort=False, rotation=270,
                      marker=dict(colors=colors, line=dict(color='#000000', width=1)))
    fig.update_layout(
        title=f'Administrative Remedy Outcomes<br>({inst_name})',
        margin={"t": 0, "b": 0, "l": 0, "r": 0},
        height=300,
    )
    # fig.update_layout(
    #     legend=dict(
    #         yanchor="top",
    #         y=0.99,
    #         xanchor="left",
    #         x=-0.21
    #     )
    # )
                    
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
    selected_subj_code_list = [subj_rows[i]['value'] for i in selected_subj_rows]

    # filter_mask = cpt_df['ITERLVL'].isin(filingSelections)
    # filter_mask &= cpt_df['cdsub1cb'].isin(selected_subj_code_list)
    filter_expr = (
            (pl.col('ITERLVL').is_in(filingSelections)) &
            (pl.col('cdsub1cb').is_in(selected_subj_code_list))
    )

    if info is None:
        inst_name = 'All Institutions'
    else:
        inst_code = info['points'][0]['customdata'][3]
        # filter_mask &= (cpt_df[trackingSelection] == inst_code)
        filter_expr &= (pl.col(trackingSelection) == inst_code)
        inst_name = name_key_df.filter(pl.col('facility_code') == inst_code)['nice_name'][0]

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
        title=f"Rolling Monthly Administrative Remedy Filings ({(inst_name)})",
        # xaxis_title="Time",
        # yaxis_title="Weekly Filing Count",
        xaxis = dict(
            rangeslider = {'visible': True,},# 'range':default_timerange},
            range = time_range if time_range!=default_timerange else default_timerange, # keep time_range permanently to keep everything 2000-2024
            autorange = False, #if time_range!=default_timerange else True, # keep false permanently to keep everything 2000-2024
            # autorangeoptions = {'minallowed':default_timerange[0], 'maxallowed':default_timerange[1]},
        ),
        margin={"t": 40, "b": 0, "l": 0, "r": 0},
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
