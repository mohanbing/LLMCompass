import json
import graphviz
from pathlib import Path


G = graphviz.Digraph(format="png", node_attr={'color': 'lightblue2', 'style': 'filled'})
path = Path("/home/amohan2/Wyvern/LLMCompass/new_tiny_decode.json")
colors = ['red', 'blue', 'green', 'purple', 'black']
color_index = 0

with open(path, 'r') as f:
    dep_graph = json.load(f)


def construct_graph(dep_graph:dict):
    input_edges = {}
    output_edges = {}
    nodes = {}

    for node, val in dep_graph.items():
        nodes[node] = []

        for _, tensor_details in val["dep"].items():
            tensor_name = tensor_details["variable_name"]
            if tensor_details["tensor_desc"] in ["K_cache", "V_cache"]:
                desc = tensor_details["tensor_desc"]
                tensor_name += f": {desc}"
            input_edges[tensor_name] = node
            nodes[node].append(tensor_name)
        
        output_tensor = val["out"]["variable_name"]
        if output_edges.get(output_tensor) is None:
            output_edges[output_tensor] = dict(src=[node], traversed=False, traversed_count=0)
        else:
            output_edges[output_tensor]["src"].append(node)


    node_to_graph_id = {}
    edge_colors = {}
    G.node('0', "Input", shape = "box", fillcolor="palegreen1")
    G.node('W', "Weights", shape = "box", fillcolor="palegreen1")
    last_node_id = -1
    color_index = 0
    for i, node in enumerate(nodes.keys()):
        op_desc = dep_graph[node]["op_desc"]
        core = dep_graph[node]["core"]
        G.node(str(i+1), f"{node}: core: {core} | {op_desc}")
        node_to_graph_id[node] = str(i+1)
        last_node_id = i+1

    last_node = None
    for node, edges in nodes.items():
        last_node = node
        for c, edge in enumerate(edges):
            if edge not in edge_colors:
                edge_colors[edge] = colors[color_index % len(colors)]
                color_index += 1

            if output_edges.get(edge) is None:
                if c == 0:
                    G.edge('0', node_to_graph_id[node], label=edge, color=edge_colors[edge])
                else:
                    G.edge('W', node_to_graph_id[node], label="Wt_"+edge, color=edge_colors[edge])
            else:
                for src in output_edges[edge]["src"]:
                    output_edges[edge]["traversed"] = True
                    G.edge(node_to_graph_id[src], node_to_graph_id[node], label=edge, color=edge_colors[edge])
                    output_edges[edge]["src"].pop(0)
    
    G.node(str(last_node_id+1), "Output", shape = "box", fillcolor="rosybrown1")
    for edge, val in output_edges.items():
        if edge not in edge_colors:
            edge_colors[edge] = colors[color_index % len(colors)]
            color_index += 1

        if val["traversed"] == False:
            for src in val["src"]:
                G.edge(node_to_graph_id[src], str(last_node_id+1), label=edge, color=edge_colors[edge])

        for src in val["src"]:
            if src == last_node:
                G.edge(node_to_graph_id[src], str(last_node_id+1), label=edge, color=edge_colors[edge])
    

construct_graph(dep_graph)
G.render("tiny")