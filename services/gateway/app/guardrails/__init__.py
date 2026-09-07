"""
Phase 5: guardrails.

Input/output moderation on both boundaries of the graph -- see
moderation.py's module docstring for what this does and does not cover, and
graph/nodes/input_guardrail.py / graph/nodes/output_guardrail.py for how it's
wired into the graph itself (not the transport layer, so both /call/turn and
/ws/call get it for free).
"""
