system_instruction = (
    "You are a helpful assistant that solves knights and knaves puzzles. "
    "Each person is either a knight (always tells the truth) or a knave (always lies). "
    "Reason step by step, then conclude with the identities in the format:\n"
    "(1) Name is a knight/knave\n"
    "(2) Name is a knight/knave\n"
    "..."
)

system_instruction_no_reason = (
    "You are a helpful assistant that solves knights and knaves puzzles. "
    "Provide only the final identities in the format:\n"
    "(1) Name is a knight/knave\n"
    "(2) Name is a knight/knave\n"
    "..."
)

demonstration_2char = (
    "### Question: A very special island is inhabited only by knights and knaves. "
    "You meet 2 inhabitants: Alice and Bob. Alice says, \"Bob is a knave.\" "
    "Bob says, \"Alice is a knight.\" So who is a knight and who is a knave?\n"
    "### Answer: Let's think step by step. "
    "If Alice were a knight, then Bob would be a knave. "
    "But then Bob's statement \"Alice is a knight\" would be false, which is consistent. "
    "If Alice were a knave, then Bob would be a knight, "
    "but then Bob's statement would be true and Alice's statement would be false, also consistent. "
    "Now check Bob: if Bob is a knave, his statement is false, so Alice is a knave, "
    "which contradicts Alice being a knight in the first case. "
    "Thus Alice is a knave and Bob is a knight. "
    "CONCLUSION: (1) Alice is a knave, (2) Bob is a knight."
)

demonstration_2char_no_reason = (
    "### Question: A very special island is inhabited only by knights and knaves. "
    "You meet 2 inhabitants: Alice and Bob. Alice says, \"Bob is a knave.\" "
    "Bob says, \"Alice is a knight.\" So who is a knight and who is a knave?\n"
    "### Answer: (1) Alice is a knave\n(2) Bob is a knight"
)
