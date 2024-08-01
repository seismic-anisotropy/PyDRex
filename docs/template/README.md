The docs template is modified from the default template found at:

<https://github.com/mitmproxy/pdoc/tree/main/pdoc/templates/default>

The most significant changes are:
- adding buttons to chose from API or tests/examples docs to the landing page
- switching to a dark CSS theme, based on the example found at the link below

<https://github.com/mitmproxy/pdoc/tree/main/examples/dark-mode>

When pdoc releases a new version we should check
if there were any upstream template/theme changes that we want to merge.
It is not necessary to copy-paste template files that we don't intend to change,
as pdoc will automatically fall back to using the files from the default theme.
