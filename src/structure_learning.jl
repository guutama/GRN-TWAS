# =============================================================================
# GRN-TWAS: Network reconstruction with BioFindr
#
# Purpose
#   Infer directed gene–gene regulatory edges using SNP instruments (eQTLs)
#   with BioFindr’s causal-inference tests.
#
# Required inputs
# 1) Expression matrix: dX :: DataFrame
#    • N rows = samples (same order as dG)
#    • G columns = gene IDs (Float64/Continuous)
#    Row   geneA      geneB      geneC      geneD      …
#    1     -0.241     0.832      -1.104     0.157      …
#    2      0.473    -0.116       0.291    -0.842      …
#    3     -1.022     0.507       0.063     0.334      …
#    …        …         …           …         …        …
#
#
# 2) Genotype matrix: dG :: DataFrame
#    • N rows = samples (same order as dX)
#    • S columns = SNP IDs (Integer 0/1/2 ))
#    Row   rs1001   rs1002   rs1003   rs1004   …
#    1       0        1        2        0      …
#    2       1        1        0        2      …
#    3       2        0        1        1      …
#    …       …        …        …        …      …
#
# 3) eQTL mapping table: dE :: DataFrame
#    • Required columns: :SNP, :gene (strings matching names in dG and dX)
#    • Optional columns (ignored by mapping): beta, t-stat, p-value, adj.p-value, …
#    Row   SNP       gene        beta       t-stat      p-value     adj.p-value
#    1     rs1001    geneA      -0.378048   -3.95352   8.79e-05      0.03698
#    2     rs1002    geneB       0.569251    6.42444   3.03e-10      0.00050
#    3     rs1003    geneC       0.321534    4.21718   2.93e-05      0.01699
#    4     rs1004    geneD       0.329905    4.08898   5.03e-05      0.04098
#    …       …         …            …           …          …            …
#

# Optional parameters
#   combination :: String ∈ ("orig","IV","mediation")
#       - "orig": BioFindr’s recommended composite of IV + relevance (robust default)
#       - "IV":   instrumental-variables (non-independence) combination
#       - "mediation": classic mediation combo
#   method      :: String ∈ ("moments","kde")  # LLR mixture fitting
#   FDR         :: Real   # q-value filter (e.g., 0.05 keeps edges with q ≤ 0.05)
#   sorted      :: Bool   # if true (default), sort by increasing q-value
#
# Output
# 4) BioFindr output: dP :: DataFrame (inferred edges)
#    • Directed regulator→target edges with posterior probability and q-value
#
#    Row   Source   Target   Probability    qvalue
#    1     geneB    geneA       0.997        0.000
#    2     geneB    geneC       0.994        0.001
#    3     geneD    geneA       0.981        0.004
#    4     geneA    geneC       0.962        0.010
#    5     geneC    geneD       0.913        0.028
#    …       …        …           …            …
#
#

# 5) BioFindr output: G

# 5-element Vector{SimpleEdge{Int64}}:
# Edge 12 => 87
# Edge 12 => 203
# Edge 45 => 107
# Edge 107 => 311
# Edge 203 => 311

# 6) BioFindr output:  name2idx 
# "geneA" => 12
# "geneB" => 87
# "geneC" => 203
# "geneD" => 45
# "geneE" => 311
# "geneF" => 107

import Pkg
Pkg.activate("/cluster/projects/nn1015k/GRN-TWAS/")

using CSV, DataFrames, BioFindr
using JLD2
using Graphs


function infer_grn(dX::DataFrame, dG::DataFrame, dE::DataFrame;
                   colX::Union{Int,Symbol}=2,
                   colG::Union{Int,Symbol}=1,
                   combination::String="orig",
                   method::String="moments",
                   FDR::Real=1.0,
                   sorted::Bool=true)




    # IMPORTANT: Samples must be aligned between dX and dG BEFORE this call.

    # --- Call BioFindr (three-argument causal inference) ---------------------
    dP = findr(dX, dG, dE; colX=colX, colG=colG,
               method=method, combination=combination, FDR=FDR, sorted=sorted)

    BioFindr.globalfdr!(dP, FDR=0.15, sorted=true)
    G, name2idx = dagfindr!(dP)
    save_path = joinpath(GRAPH_DIR, "grn.jld2")
    JLD2.@save save_path dP G name2idx

    return G,name2idx

end



