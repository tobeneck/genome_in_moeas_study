# This file contains some methods to assist with the plots

import plotly.graph_objects as go


def set_margins(fig:go.Figure, left=0, right=0, top=25, bottom=0) -> go.Figure:
    fig.update_layout(
        margins=dict(
            l=left,
            r=right,
            t=top,
            b=bottom,
        )
    )
    return fig
