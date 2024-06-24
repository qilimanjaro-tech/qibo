def _get_style(style_name):

    if "garnacha" in style_name:
        return {
            "facecolor": "#5e2129",
            "edgecolor": "#ffffff",
            "linecolor": "#ffffff",
            "textcolor": "#ffffff",
            "fillcolor": "#ffffff",
            "gatecolor": "#5e2129",
            "controlcolor": "#ffffff",
        }

    if "fardelejo" in style_name:
        return {
            "facecolor": "#e17a02",
            "edgecolor": "#fef1e2",
            "linecolor": "#fef1e2",
            "textcolor": "#FFFFFF",
            "fillcolor": "#fef1e2",
            "gatecolor": "#8b4513",
            "controlcolor": "#fef1e2",
        }

    if "quantumspain" in style_name:
        return {
            "facecolor": "#EDEDF4",
            "edgecolor": "#092D4E",
            "linecolor": "#092D4E",
            "textcolor": "#8561C3",
            "fillcolor": "#092D4E",
            "gatecolor": "#53E7CA",
            "controlcolor": "#092D4E",
        }

    if "color-blind" in style_name:
        return {
            "facecolor": "#d55e00",
            "edgecolor": "#f0e442",
            "linecolor": "#f0e442",
            "textcolor": "#f0e442",
            "fillcolor": "#cc79a7",
            "gatecolor": "#d55e00",
            "controlcolor": "#f0e442",
        }

    return {
        "facecolor": "w",
        "edgecolor": "#000000",
        "linecolor": "k",
        "textcolor": "k",
        "fillcolor": "#000000",
        "gatecolor": "w",
        "controlcolor": "#000000",
    }
