Welcome to [The Remedy Project](https://theremedyproj.org)'s Administrative Remedy Dashboard! This pop-up will be your guide to the dashboard. If you ever want to revisit it, you can click the info button in the bottom right of your screen.

This webpage houses a visualization of historical data from the Federal Bureau of Prisons (FBOP) on the outcomes of complaints and requests made by individuals incarcerated in FBOP facilities. These complaints/requests are made in the form of filings known as Administrative Remedies, which are at the core of The Remedy Project’s [advocacy work](https://www.theremedyproj.org/our-impact). Administrative Remedies are the *only* official avenue for people incarcerated in the FBOP to raise an issue or seek relief for the conditions of their confinement.

The data summarized on this page cover nearly 25 years worth (01/2000 - 06/2024) of Administrative Remedy filing outcomes from FBOP facilities, comprising 1.78 million individual records. Each record represents the outcome of a single complaint/request (note that some records may represent a resubmission of an earlier filing).

Filings proceed through 4 stages, where each subsequent stage is an appeal of the previous stage:
*   BP8 form: Initial request for informal resolution
*   BP9 form: Formal Administrative Remedy request to the facility’s warden
*   BP10 form: Appeal to the Regional Director
*   BP11 form: Appeal to the Agency's General Counsel

For a more detailed description of the Administrative Remedy process, please refer to this [info sheet](https://cic.dc.gov/sites/default/files/dc/sites/cic/page_content/attachments/BOP%20Administrative%20Remedies%2011.15.17%20REVISED.pdf) and this [guide](https://www.washlaw.org/pdf/BOP_Grievance_Guide.pdf). Additional details, including specifics on FBOP policies and examples of filings, can be found [here](https://www.law.umich.edu/facultyhome/margoschlanger/Pages/PrisonGrievanceProceduresandSamples.aspx).

Informal resolutions are not tracked by the FBOP, and therefore, only the formal Administrative Remedy process—the final 3 stages of BP9 through BP11—is represented in this data. As each record corresponds to a single complaint/request, this means that each documented outcome stems from the highest stage of the appeal process reached. The default is to view cases by their institution of origin; however, if you wish to view them by the institution responsible for their final outcome, you can select to track cases by the “Office Responsible for Outcome.” You can also filter by the various filing levels using the checkboxes at top left, and by case subject by selecting from the table at top right.

The geographical distribution of cases can be explored in the map by clicking and dragging to navigate, or by scrolling to zoom in and out. Larger points correspond to locations with a higher number of case filings, and darker points correspond to a higher non-approval rate. Hovering over individual points will provide information specific to that institution, and will update the other plots accordingly. If you wish to explore an institution in detail, clicking on it will select that institution for the entire page until you click somewhere else in the map.

The timeline at the bottom shows the rolling monthly case total as a function of time, where this tracks the date of the initial BP9 filing associated with the request/complaint. You can zoom into portions of the timeline by clicking and dragging across the main plot, or adjusting the edges of the smaller plot below it. This too will update the other plots accordingly. To zoom back out to the full window, just double click anywhere on the main plot.

Lastly, the case outcomes are shown in a pie chart at right. Hovering over each piece of the pie will provide detailed statistics of its contents. Cases with the “Closed (Other)” distinction may have been closed for several reasons, with the possible options being listed [here](https://docs.google.com/document/d/1vTuyUFNqS9tex4_s4PgmhF8RTvTb-uFMN5ElDjjVHTM/edit?tab=t.0#heading=h.m9dnvmnc5wti). Cases with the “Accepted” distinction have been filtered from the dataset, as they represent cases that have not yet reached an outcome.

______

We are extremely grateful to the [Data Liberation Project](https://www.data-liberation-project.org/), now [MuckRock](https://www.muckrock.com/), for their work requesting, compiling, and cleaning this dataset, which is publicly available at [this link](https://www.data-liberation-project.org/datasets/federal-inmate-complaints/). The code to produce this data visualization was written in [Plotly Dash](https://dash.plotly.com/) for Python, and is publicly available [here](https://github.com/The-Remedy-Project/data_dashboard).