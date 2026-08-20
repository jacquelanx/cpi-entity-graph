"""
`graph` -- the DETERMINISTIC stage: detected spans in, knowledge graph out.

Package marker. The spine is at the top level (`models`, `loader`, `pipeline`,
`serialize`) and the work is in the subpackages:

    text/         transcript utilities with no graph knowledge
    rules/        the rule layer -- one module per inference stage
    checks/       deterministic checkers -- one module per field checked
    second_line/  THE arbitration point: rule vs LLM, per field

Start at `pipeline.run_pipeline`; it names every stage in the order they run.
"""
