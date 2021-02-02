using UniversalPricing
using Documenter

makedocs(;
    modules=[UniversalPricing],
    authors="SciQuant",
    repo="https://github.com/rvignolo/UniversalPricing.jl/blob/{commit}{path}#L{line}",
    sitename="UniversalPricing.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
