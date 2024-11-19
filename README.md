This is the repository for The Remedy Project Bureau of Prison Administrative Remedy Data Dashboard. Code is written in Python with [Plotly Dash](https://dash.plotly.com/).

# Local Development
Clone the repository to your local machine, and install the contents of the `requirements.txt` file using pip or conda (recommended to do so in a virtual environment with python 3.10).

Inside the virtual environment, to start the web app, run (from the repo's base directory): 
```
python src/data_dashboard.py
```

Then navigate to [http://localhost:8051/](http://localhost:8051/) in your local browser.

To access the [dev tools suite](https://dash.plotly.com/devtools), you will want to set `Debug=True` in the _main_ block of the code in `src/data_dashboard.py`. Note that this will result in an additional process being initialized that will result in additional memory consumption.
