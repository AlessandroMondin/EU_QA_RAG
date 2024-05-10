rag_template = """
The user asked a <<QUESTION>>. Use <<DOCUMENTS>> to effectively reply
to the question of the user. You cannot get an answer from <<EXAMPLES>>,
tell the user that that information cannot be retrieved within the documents
and that he should research it with other tools. Your must reply in at most
100 words, your answer must be explicative but concise.

<<DOCUMENTS>>:
{documents}

<<MESSAGE>>
{message}

<<OUTPUT>>
"""
