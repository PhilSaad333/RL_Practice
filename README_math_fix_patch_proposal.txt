Summary:
- Normalize the README math blocks so GitHub renders them without MathJax parse errors.

Rationale:
- Display equations that are indented inside list items confuse GitHub's parser; removing the indentation is the recommended workaround.
- Simplifying the notation around the approximations (replacing `\rightarrow` with `\approx`) avoids unmatched-brace errors in GitHub's MathJax build.
- Cleaning up the entropy-to-go expectation formula ensures the block renders and matches the docstring notation elsewhere.

Minimal Diffs:
- README.md:77 remove the leading spaces before each display-math fence starting with `$$` and ending with `$$`.
- README.md:80 rewrite the per-token entropy estimator to use `\approx` and align the denominator terms with braces that MathJax accepts on GitHub (`\left(\sum ... \right)` instead of bare parentheses).
- README.md:95 drop the `\!` spacing command and leave `\mathbb{E}_{t \sim \pi}\left[\sum_k H_k\right]` to avoid MathJax complaints.
- README.md:109 expand `\delta \mathcal{H}_1/\eta` into `\frac{\delta \mathcal{H}_1}{\eta}` so the fraction braces are explicit.
- README.md:124 switch the final equation to use `\mid` instead of `|` inside the conditional probability and remove redundant spaces.

Steps:
1. Update each affected math fence so the `$$` delimiters are flush-left and each block has a blank line before/after.
2. Apply the notation cleanups in the four equations called out above.
3. Scan the surrounding paragraphs to ensure list numbering and indentation still look correct.

Validation:
- Open the README in a GitHub-flavored Markdown preview to confirm all math blocks render without warnings.
- Spot-check that the inline references to the updated notation (entropy-to-go, control variates) still match the equations.

Rollback:
- Revert README.md to the previous commit if GitHub rendering still fails or any notation regression is spotted.
