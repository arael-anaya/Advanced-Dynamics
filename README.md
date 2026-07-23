# Advanced Dynamics — LaTeX setup

This repo tracks both the `.tex` sources and their compiled `.pdf` output. To
build the PDFs locally you need a LaTeX distribution installed.

## One-time install (Ubuntu/Debian)

```sh
sudo apt update
sudo apt install -y texlive texlive-latex-extra texlive-fonts-extra texlive-science latexmk
```

This covers every package used across the homework files (`tcolorbox`,
`siunitx`, `mathtools`, `booktabs`, `hyperref`, etc.) plus `latexmk`, which
drives the builds below. (`texlive-full` also works but is several GB larger
and installs language packs and extras this repo doesn't use.)

If you use VS Code, the **LaTeX Workshop** extension
(`James-Yu.latex-workshop`) gives in-editor build/preview on top of the same
install.

## Building

Build everything:

```sh
./build.sh
```

Build one homework only:

```sh
./build.sh "Homework 3"
```

Clean up latexmk build artifacts (aux/log/etc.), everywhere:

```sh
./build.sh --clean
```

Or run `latexmk -pdf` directly from inside any folder containing a `.tex` file.

## Automatic build on commit

This repo ships a pre-commit hook (`.githooks/pre-commit`) that recompiles
any `.tex` file you've staged and stages the resulting `.pdf` alongside it,
so a committed PDF always matches the source it was built from. It also
blocks the commit if a file fails to compile.

The hook is enabled for this clone via:

```sh
git config core.hooksPath .githooks
```

Run that once on any new clone (already done on this machine). To skip it
for a single commit, use `git commit --no-verify`.
