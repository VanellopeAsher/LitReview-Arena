"""
Structure extraction, graph building, and gap anchors (LitJudge Section 6.1).

Separated from utils.py so `utils.py` can host the user's LLM client only.
"""

import re
from typing import List, Tuple
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    nx = None


def extract_skeleton_text(text: str) -> Tuple[List[str], List[str]]:
    section_headers = []
    paragraph_leads = []

    header_pattern = r'^#{1,6}\s+(.+)$'
    lines = text.split('\n')
    in_paragraph = False

    for line in lines:
        stripped = line.strip()
        header_match = re.match(header_pattern, stripped)
        if header_match:
            in_paragraph = False
            header_text = header_match.group(1).strip()
            section_headers.append(header_text)
        elif stripped and not stripped.startswith('-') and not stripped.startswith('*'):
            if not in_paragraph:
                sentences = re.split(r'[.!?]\s+', stripped)
                if sentences and sentences[0]:
                    lead = sentences[0][:100].strip()
                    if lead:
                        paragraph_leads.append(lead)
                in_paragraph = True
        else:
            in_paragraph = False

    return section_headers, paragraph_leads


def build_paragraph_network(section_headers: List[str], paragraph_leads: List[str]):
    if nx is None:
        class SimpleGraph:
            def __init__(self):
                self.nodes_data = {}
                self.edges = []

            def add_node(self, node, **kwargs):
                self.nodes_data[node] = kwargs

            def add_edge(self, u, v, **kwargs):
                self.edges.append((u, v, kwargs))

        G = SimpleGraph()
    else:
        G = nx.DiGraph()

    for i, header in enumerate(section_headers):
        G.add_node(f"section_{i}", type="section", text=header)

    for i, lead in enumerate(paragraph_leads):
        G.add_node(f"para_{i}", type="paragraph", text=lead)

    for i in range(len(section_headers) - 1):
        G.add_edge(f"section_{i}", f"section_{i+1}", relation="sequential")

    if paragraph_leads:
        for i, header in enumerate(section_headers):
            if i < len(paragraph_leads):
                G.add_edge(f"section_{i}", f"para_{i}", relation="contains")

    return G


def compute_graph_similarity(G1, G2) -> float:
    def get_node_type_dist(G):
        type_dist = defaultdict(int)
        if hasattr(G, 'nodes'):
            nodes = G.nodes()
        elif hasattr(G, 'nodes_data'):
            nodes = G.nodes_data.keys()
        else:
            return {}

        for node in nodes:
            if hasattr(G, 'nodes'):
                node_type = G.nodes[node].get('type', 'unknown')
            elif hasattr(G, 'nodes_data'):
                node_type = G.nodes_data[node].get('type', 'unknown')
            else:
                node_type = 'unknown'
            type_dist[node_type] += 1
        total = sum(type_dist.values())
        return {k: v / total if total > 0 else 0 for k, v in type_dist.items()}

    dist1 = get_node_type_dist(G1)
    dist2 = get_node_type_dist(G2)

    types = set(dist1.keys()) | set(dist2.keys())
    if not types:
        return 0.0

    intersection = sum(min(dist1.get(t, 0), dist2.get(t, 0)) for t in types)
    union = sum(max(dist1.get(t, 0), dist2.get(t, 0)) for t in types)

    if union == 0:
        return 0.0

    return intersection / union


def extract_gap_anchors(text: str) -> List[str]:
    anchors = []

    gap_patterns = [
        r'(?:future\s+work|future\s+directions?|research\s+gaps?|limitations?|challenges?|opportunities?)',
        r'(?:gaps?\s+in|directions?\s+for|next\s+steps?|open\s+questions?)'
    ]

    lines = text.split('\n')
    in_gap_section = False

    for line in lines:
        line_lower = line.lower()

        if any(re.search(pattern, line_lower) for pattern in gap_patterns):
            in_gap_section = True
            continue

        if in_gap_section:
            bullet_match = re.match(r'^[-*]\s+(.+)$', line.strip())
            if bullet_match:
                anchor = bullet_match.group(1).strip()
                anchor = re.sub(r'\[.*?\]', '', anchor)
                anchor = anchor[:200].strip()
                if anchor:
                    anchors.append(anchor)
            elif re.match(r'^#{1,6}\s+', line.strip()):
                in_gap_section = False

    return anchors[:10]
