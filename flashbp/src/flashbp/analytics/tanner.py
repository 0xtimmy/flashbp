import math
import pydot


def plot_tanner_graph(decoder, output_path="tanner.png", dpi=None):
    """
    Render the Tanner graph of a FlashBP decoder as a PNG using pydot/graphviz.

    Variable nodes (error mechanisms) are drawn as circles.
    Check nodes (detectors) are drawn as squares.
    An edge connects check d to variable e wherever H[d, e] = 1.

    Parameters
    ----------
    dpi : int | None
        Output resolution in dots per inch.  Defaults to 72 * sqrt(num_errors),
        so larger codes produce proportionally larger images.
    """
    effective_dpi = dpi if dpi is not None else int(72 * math.sqrt(decoder.num_errors))

    graph = pydot.Dot("tanner_graph", graph_type="graph", rankdir="LR",
                      dpi=effective_dpi)
    graph.set_node_defaults(fontname="Helvetica", fontsize="10")

    # Variable nodes — one per error mechanism
    for e in range(decoder.num_errors):
        node = pydot.Node(
            f"v{e}",
            label=f"e{e}",
            shape="circle",
            style="filled",
            fillcolor="#AED6F1",
            width="0.4",
            height="0.4",
            fixedsize="true",
        )
        graph.add_node(node)

    # Check nodes — one per detector
    for d in range(decoder.num_detectors):
        node = pydot.Node(
            f"c{d}",
            label=f"d{d}",
            shape="square",
            style="filled",
            fillcolor="#A9DFBF",
            width="0.4",
            height="0.4",
            fixedsize="true",
        )
        graph.add_node(node)

    # Edges from H
    H = decoder.H
    for d in range(decoder.num_detectors):
        for e in range(decoder.num_errors):
            if H[d, e]:
                graph.add_edge(pydot.Edge(f"c{d}", f"v{e}", color="#555555"))

    graph.write_png(output_path)
    print(f"Tanner graph saved to {output_path}")
    return graph
