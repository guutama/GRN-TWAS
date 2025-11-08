# %%
using Pkg: Pkg
Pkg.activate("/cluster/projects/nn1015k/GRN-TWAS/")


# %%
using CSV, DataFrames, ThreadsX, StatsBase, BioFindr, Glob, Graphs, FilePathsBase, Plots, MLJLinearModels,
    MLJBase, Optim, Statistics, LinearAlgebra, ProgressMeter, MLUtils, MLJModelInterface,
    MLJTuning, Random, MLJModels, Distributions, PrettyTables, MultipleTesting, Distances, MLJ





# %%
# ╔═╡ f6186acf-ceb9-47c4-a359-b1f1bbed5979
using Measures

# ╔═╡ 423197c6-8d2b-490f-be83-79f258a1feac
using JLD2: JLD2
using MLJScikitLearnInterface: MLJScikitLearnInterface


# %%
RidgeRegressorSK = MLJ.@load RidgeRegressor pkg = MLJScikitLearnInterface
MLJ.@load BayesianRidgeRegressor pkg = MLJScikitLearnInterface
RidgeRegressorMLJ = @load RidgeRegressor pkg = MLJLinearModels
LinearRegressor = @load LinearRegressor pkg = MLJLinearModels

# %%
begin
    BASE_DIR = "/cluster/projects/nn1015k/GRN-TWAS/"
    DATA_DIR = joinpath(BASE_DIR, "data")
    MODEL_DIR = joinpath(DATA_DIR, "model")
    INFO_DIR = joinpath(DATA_DIR, "info")
    GWAS_DIR = joinpath(DATA_DIR, "gwas")
    PREPROCESSED = joinpath(DATA_DIR, "preprocessed")
    GRAPH_PATH = joinpath(DATA_DIR, "graph")
end


# %%
# geno_path = joinpath(PREPROCESSED, "all_geno_original.jld2")

# # ╔═╡ 3f1c2cad-34bc-4ac2-8a58-880c9b5bb3cc
# JLD2.@load geno_path filtered_original_dfs

# %%
all_expression_path = joinpath(PREPROCESSED, "all_exp.jld2")
JLD2.@load all_expression_path common_sample_expression_dfs

# %%
tissues = ["AOR", "Blood", "LIV", "MAM", "SKLM", "SF", "VAF"] #"Blood", "LIV", "MAM", "SKLM", "SF", "VAF"

# %%
transposed_exp = ThreadsX.map(tissues) do tissue
    df = deepcopy(common_sample_expression_dfs[tissue])
    #extract new column names for transposed data (genes)
    new_column_names = Symbol.(df[!, :id])

    # extract numeric values
    value_df = DataFrames.select(df, Not(:id))

    transposed = transpose(Matrix(value_df))

    df_t = DataFrame(transposed, new_column_names)

    sample_names = names(value_df)

    insertcols!(df_t, 1, :sample => sample_names)
    return tissue => df_t
end

# ╔═╡ 9917a179-daf4-4446-a770-6331d3142638
exp_transposed = Dict(transposed_exp)

# %%



# %%
using JSON3

# %%


# Dictionary to hold everything
results_per_tissue = Dict{String,Dict{String,Any}}()

for tissue in tissues
    file_path = joinpath(MODEL_DIR, "metrices_and_params_gp$(tissue).json")
    if isfile(file_path)
        results_per_tissue[tissue] = JSON3.read(open(file_path, "r"), Dict{String,Any})
    else
        @warn "File for $tissue not found at $file_path"
    end
end




# %%
results_per_tissue["Blood"]

# %%
gwas_file = joinpath(GWAS_DIR, "gwas_summary_extracted_mapped_final.tsv")
all_gwas = CSV.read(gwas_file, DataFrame, delim='\t')
# gwas = all_gwas[:, [:snpID, :beta, :standard_error]]


# %%
# first(all_gwas,5)

# %%
function allele_qc!(effect_allele, other_allele, ref, alt; complements=Dict("A" => "T", "T" => "A", "C" => "G", "G" => "C"))
    n = length(effect_allele)
    @assert n == length(ref) == length(alt) == length(other_allele) "Input vectors must be the same length."

    # Pre-allocate output masks
    keep = trues(n)
    flip = falses(n)

    # Pre-compute flipped alleles only once
    flip1 = [get(complements, r, r) for r in ref]
    flip2 = [get(complements, r, r) for r in alt]

    @inbounds @simd for i in 1:n
        ea = uppercase(effect_allele[i])
        oa = uppercase(other_allele[i])
        r1 = uppercase(ref[i])
        r2 = uppercase(alt[i])

        # Exclude ambiguous SNPs
        ambiguous = (ea == "A" && oa == "T") || (ea == "T" && oa == "A") ||
                    (ea == "C" && oa == "G") || (ea == "G" && oa == "C")

        valid = (ea in ("A", "T", "C", "G")) && (oa in ("A", "T", "C", "G"))

        if ambiguous || !valid
            keep[i] = false
        end

        # Determine if flipping is required
        if (ea == r2 && oa == r1) || (ea == flip2[i] && oa == flip1[i])
            flip[i] = true
        end
    end

    return Dict("keep" => keep, "flip" => flip)
end



# %%

allele_allignment_result = ThreadsX.map(tissues) do tissue

    gwas_df = all_gwas
    eqtl_df = filtered_original_dfs[tissue]

    # 1. Find Common SNPs and Create Fast Lookup
    common_snps = intersect(gwas_df.snpID, eqtl_df.rs_id)
    common_snps_set = Set(common_snps)

    gwas_df_aligned = gwas_df[in.(gwas_df.snpID, Ref(common_snps_set)), :]
    eqtl_df_aligned = eqtl_df[in.(eqtl_df.rs_id, Ref(common_snps_set)), :]

    # 2. Perform Explicit Join for Perfect Alignment
    aligned_df = innerjoin(
        gwas_df_aligned,
        eqtl_df_aligned,
        on=[:snpID => :rs_id],
        makeunique=true  # Resolve duplicate column names automatically
    )

    # 3. Apply Allele QC
    qc_result = allele_qc!(
        aligned_df.effect_allele,
        aligned_df.other_allele,
        aligned_df.ref,
        aligned_df.alt,
    )

    keep_mask = qc_result["keep"]
    flip_mask = qc_result["flip"]
    # 4. Filter Final Data
    final_df = aligned_df[keep_mask, :]
    # 5. Flip Alleles in Final EQTL Data

    # Determine the correct column names after join (in case they were renamed)
    col_names = names(final_df)

    ref_col = findfirst(name -> occursin("ref", String(name)), col_names)
    alt_col = findfirst(name -> occursin("alt", String(name)), col_names)
    @assert !isnothing(ref_col) "Reference allele column not found!"
    @assert !isnothing(alt_col) "Alternate allele column not found!"
    ref_col_name = col_names[ref_col]
    alt_col_name = col_names[alt_col]
    # Access the columns
    ref_vals = final_df[!, ref_col_name]
    alt_vals = final_df[!, alt_col_name]
    # Compute the final flip mask after filtering with keep_mask
    flip_submask = flip_mask[keep_mask]

    # Flip alleles in-place
    flip_indices = findall(flip_submask)  # Compute indices only once

    for i in flip_indices
        ref_vals[i], alt_vals[i] = alt_vals[i], ref_vals[i]
    end


    # 6. Flip GWAS Z-scores or Beta Values

    if :Z in propertynames(final_df)
        final_df[flip_submask, :Z] .= -final_df[flip_submask, :Z]
    elseif :beta in propertynames(final_df)
        final_df[flip_submask, :beta] .= -final_df[flip_submask, :beta]
    else
        @warn "No 'Z' or 'beta' column found in the final DataFrame for flipping!"
    end

    # 7. Compute and Add Z-scores if not already present
    if :Z ∉ propertynames(final_df)
        if (:beta in propertynames(final_df)) && (:standard_error in propertynames(final_df))
            β = final_df[:, :beta]
            se = final_df[:, :standard_error]
            # Compute Z-scores safely, handle division by zero or missing SE values
            z_scores = similar(β)
            for i in eachindex(β)
                z_scores[i] = isfinite(se[i]) && se[i] != 0 ? β[i] / se[i] : missing
            end
            final_df[!, :Z] = z_scores
        else
            @warn "Cannot compute Z-scores: 'beta' or 'standard_error' column missing."
        end
    end
    return tissue => final_df
end


# %%
allele_alignment_dict = Dict(allele_allignment_result)


# %%


transposed_genotype_results = ThreadsX.map(tissues) do tissue
    final_df = allele_alignment_dict[tissue]  # Your pre-aligned DataFrame for the tissue

    # Dynamically detect genotype columns starting with "<tissue>_"
    prefix = tissue * "_"
    geno_columns = filter(name -> startswith(String(name), prefix), names(final_df))

    # Extract and transpose genotype data
    geno_df = DataFrames.select(final_df, geno_columns)
    transposed = transpose(Matrix(geno_df))

    # Construct DataFrame with SNP names as columns and sample names as a column
    snp_names = final_df.snpID  # Ensure this column exists in final_df
    # df_t = DataFrame(transposed, Symbol.(snp_names); makeunique=true)
    unique_names = unique(snp_names)
    df_t = DataFrame(transposed[:, 1:length(unique_names)], Symbol.(unique_names))

    sample_names = names(geno_df)
    insertcols!(df_t, 1, :sample => sample_names)

    # Return as Pair to construct a dictionary later
    return tissue => df_t
end

# Convert to Dict if needed
transposed_genotype_dict = Dict(transposed_genotype_results...)

# %%
# z_gwas = compute_gwas_zscores(gwas_summary)

# %%


# %%
function standardize_df(df::DataFrame)
    X_mat = Matrix(df)
    means = mean(X_mat, dims=1)
    stds = std(X_mat, dims=1)
    standardized = (X_mat .- means) ./ stds
    return DataFrame(standardized, names(df)), means, stds
end
function coerce_to_continuous(df::DataFrame)
    return coerce(df, Dict(Symbol(c) => MLJModelInterface.Continuous for c in names(df)))
end


# %%
function pvalue_z_safe(z::Real; tails::Symbol=:two)
    dist = Distributions.Normal(0, 1)
    if tails == :two
        return 2 * exp(logccdf(dist, abs(z)))
    elseif tails == :right
        return exp(logccdf(dist, z))
    elseif tails == :left
        return exp(logcdf(dist, z))
    else
        throw(ArgumentError("Invalid option for tails. Use :two, :right, or :left."))
    end
end


# %%
# Pkg.add("StatsFuns")

# %%
using StatsFuns



# then if you really need p‐values, just exp(logpvalue_z(z))…


# %%

# — 1) Extract cis/trans weight vectors from  model result dict
function extract_weights_and_features(result::Dict{String,Any})
    cis_feats = cis_coefs = nothing
    trans_feats = trans_coefs = nothing

    mode = get(result, "mode", "")
    params = get(result, "params", Dict{String,Any}())

    if mode == "cis_only"
        cis_feats = collect(keys(params))
        cis_coefs = collect(values(params))

    elseif mode == "trans_only"
        trans_feats = collect(keys(params))
        trans_coefs = collect(values(params))

    elseif mode == "cis_trans"
        if haskey(params, "cis")
            c = params["cis"]
            cis_feats = collect(keys(c))
            cis_coefs = collect(values(c))
        end
        if haskey(params, "trans")
            t = params["trans"]
            trans_feats = collect(keys(t))
            trans_coefs = collect(values(t))
        end
    end

    return (
        cis=(features=cis_feats, coefs=cis_coefs),
        trans=(features=trans_feats, coefs=trans_coefs)
    )
end

# — 2) Compute p‐value from Z in log‐space
function logpvalue_z(z::Real; tails::Symbol=:two)
    dist = Normal(0, 1)
    if tails == :two
        return log(2) + logccdf(dist, abs(z))
    elseif tails == :right
        return logccdf(dist, z)
    elseif tails == :left
        return logcdf(dist, z)
    else
        throw(ArgumentError("Invalid tails: $tails"))
    end
end

function compute_Z_score(α::AbstractVector{<:Real},
    z_g::AbstractVector{<:Real},
    geno_df::AbstractDataFrame,
    y::AbstractVector{<:Real};
    r2_threshold::Float64=1e-3)
    try
        X = Matrix(geno_df)
        n_feats = size(X, 2)

        if n_feats == 1
            x = X[:, 1]
            σ2_x = var(x)
            σ_g2 = α[1]^2 * σ2_x
            r2 = σ_g2 / var(y)
            if r2 < r2_threshold
                return (nothing, σ_g2, r2)
            end
            σ_s = std(x)
            σ_g = sqrt(σ_g2)
            Z = α[1] * (σ_s / σ_g) * z_g[1]
            return (Z, σ_g2, r2)
        end

        # multi‐SNP path
        Xc = X .- mean(X; dims=1)
        σ_g2 = var(Xc * α)
        r2 = σ_g2 / var(y)
        if r2 < r2_threshold
            return (nothing, σ_g2, r2)
        end
        σ_s = std.(eachcol(Xc))
        σ_g = sqrt(σ_g2)
        Z = sum(α .* (σ_s ./ σ_g) .* z_g)
        return (Z, σ_g2, r2)

    catch e
        @warn "compute_Z_score failed: $e"
        return (nothing, nothing, nothing)
    end
end




# — 4) Placeholder push for missing results
function push_placeholder!(acc, gene::String, result::Dict{String,Any})
    push!(acc, gene => Dict(
        "Z_score" => Dict("cis" => nothing, "trans" => nothing, "combined" => nothing),
        "pval" => Dict("cis" => nothing, "trans" => nothing, "combined" => nothing),
        "sigma2" => Dict("cis" => nothing, "trans" => nothing, "combined" => nothing),
        "r2" => Dict("cis" => nothing, "trans" => nothing, "combined" => nothing),
        "feature_size" => Dict("cis" => 0, "trans" => 0, "combined" => 0),
        "mode" => get(result, "mode", "NA"),
        "metrics" => get(result, "metrics", nothing),
        "feats_cis" => String[],
        "feats_trans" => String[],
        "weights" => get(get(result, "params", Dict()), "weights", nothing)
    ))
end

# %%


using DataFrames, Statistics, Distributions, Logging

function compute_Z_score(α::AbstractVector{<:Real},
    z_g::AbstractVector{<:Real},
    geno_df::AbstractDataFrame,
    y::AbstractVector{<:Real};
    r2_threshold::Float64=1e-3)

    try
        X = Matrix(geno_df)
        n_feats = size(X, 2)

        # --- EARLY GUARD FOR ZERO FEATURES ---
        if n_feats == 0
            # no SNPs → zero variance → Z undefined
            return (nothing, 0.0, 0.0)
        end

        # --- Single‐SNP shortcut ---
        if n_feats == 1
            x = X[:, 1]
            σ2_x = var(x)                    # sample var
            σ_g2 = α[1]^2 * σ2_x
            r2 = σ_g2 / var(y)
            if r2 < r2_threshold
                return (nothing, σ_g2, r2)
            end
            σ_s = std(x)
            σ_g = sqrt(σ_g2)
            Z = α[1] * (σ_s / σ_g) * z_g[1]
            return (Z, σ_g2, r2)
        end

        # --- Multi‐SNP path ---
        Xc = X .- mean(X; dims=1)
        σ_g2 = var(Xc * α)               # var of predicted value
        r2 = σ_g2 / var(y)
        if r2 < r2_threshold
            return (nothing, σ_g2, r2)
        end
        σ_s = std.(eachcol(Xc))
        σ_g = sqrt(σ_g2)
        Z = sum(α .* (σ_s ./ σ_g) .* z_g)
        return (Z, σ_g2, r2)

    catch e
        # debug logging
        @warn "compute_Z_score failed" error = e
        # α_len=length(α) z_g_len=length(z_g)
        #       geno_df_shape=size(geno_df) y_len=length(y)
        return (nothing, nothing, nothing)
    end
end



# — 2) a placeholder for when no valid Z can be computed —
function push_placeholder!(acc, gene, result)
    push!(acc, gene => Dict(
        "Z_score" => Dict("cis" => nothing, "trans" => nothing, "combined" => nothing),
        "pval" => Dict("cis" => nothing, "trans" => nothing, "combined" => nothing),
        "sigma2" => Dict("cis" => nothing, "trans" => nothing, "combined" => nothing),
        "r2" => Dict("cis" => nothing, "trans" => nothing, "combined" => nothing),
        "feature_size" => Dict("cis" => 0, "trans" => 0, "combined" => 0),
        "mode" => get(result, "mode", "NA"),
        "metrics" => get(result, "metrics", nothing),
        "feats_cis" => String[],
        "feats_trans" => String[],
        "weights" => get(get(result, "params", Dict()), "weights", nothing)
    ))
end

# — 3) process a single tissue with up to 5 genes for testing —
function process_one_tissue(tissue,
    results_per_tissue,
    transposed_genotype_dict,
    exp_transposed,
    allele_alignment_dict)

    model_data = get(results_per_tissue, tissue, missing)
    geno = get(transposed_genotype_dict, tissue, missing)
    exp_df = get(exp_transposed, tissue, missing)
    df = allele_alignment_dict[tissue]
    zmap = Dict(r.snpID => r.Z for r in eachrow(df))
    genes = model_data === missing ? String[] : collect(keys(model_data))

    acc = Vector{Pair{String,Dict{String,Any}}}()

    if model_data === missing || geno === missing || isempty(genes)
        @warn "Skipping $tissue: missing data"
        for g in genes
            push_placeholder!(acc, g, Dict("mode" => "NA", "metrics" => Dict(), "params" => Dict()))
        end
        return tissue => Dict(acc)
    end

    for g in genes#[1:min(5, end)]
        result = model_data[g]
        mode = result["mode"]
        # display("Processing gene: $g in tissue: $tissue with mode: $mode")
        y = Vector{Float64}(exp_df[:, g])
        cis, trans = extract_weights_and_features(result)

        feats_c = cis.features === nothing ? String[] : filter(s -> haskey(zmap, s), cis.features)
        feats_t = trans.features === nothing ? String[] : filter(s -> haskey(zmap, s), trans.features)

        # initialize
        Zc = Zt = Zcomb = nothing
        σ2c = σ2t = σ2comb = nothing
        r2c = r2t = r2comb = nothing
        nc = nt = ncomb = 0

        # cis_only
        if mode == "cis_only" && !isempty(feats_c)
            α = Float64[cis.coefs[i] for (i, s) in enumerate(cis.features) if s in feats_c]
            z = Float64[zmap[s] for s in feats_c]
            dfc = DataFrames.select(geno, feats_c...)   # use the splatted syntax!
            Zc, σ2c, r2c = compute_Z_score(α, z, dfc, y)
            Zcomb, σ2comb, r2comb = Zc, σ2c, r2c
            nc = length(feats_c)
        end

        # trans_only
        if mode == "trans_only" && !isempty(feats_t)
            α = Float64[trans.coefs[i] for (i, s) in enumerate(trans.features) if s in feats_t]
            z = Float64[zmap[s] for s in feats_t]
            dft = DataFrames.select(geno, feats_t...)
            Zt, σ2t, r2t = compute_Z_score(α, z, dft, y)
            Zcomb, σ2comb, r2comb = Zt, σ2t, r2t
            nt = length(feats_t)
        end

        # cis_trans
        if mode == "cis_trans"
            αc_sel = Float64[cis.coefs[i] for (i, s) in enumerate(cis.features) if s in feats_c]
            αt_sel = Float64[trans.coefs[i] for (i, s) in enumerate(trans.features) if s in feats_t]
            z_c = Float64[zmap[s] for s in feats_c]
            z_t = Float64[zmap[s] for s in feats_t]
            w = result["params"]["weights"]
            αc_w = w["cis"] .* αc_sel
            αt_w = w["trans"] .* αt_sel

            comb = vcat(feats_c, feats_t)
            if !isempty(comb)
                α_comb = vcat(αc_w, αt_w)
                z_comb = vcat(z_c, z_t)
                dfc = DataFrames.select(geno, feats_c...)
                dft = DataFrames.select(geno, feats_t...)
                dfcomb = DataFrames.select(geno, comb...)
                Zcomb, σ2comb, r2comb = compute_Z_score(α_comb, z_comb, dfcomb, y)
                Zc, σ2c, r2c = compute_Z_score(αc_w, z_c, dfc, y)
                Zt, σ2t, r2t = compute_Z_score(αt_w, z_t, dft, y)
            end
            nc, nt, ncomb = length(feats_c), length(feats_t), length(comb)
        end

        # if no valid Z, push placeholder
        if (mode == "cis_only" && Zc === nothing) ||
           (mode == "trans_only" && Zt === nothing) ||
           (mode == "cis_trans" && Zcomb === nothing)
            push_placeholder!(acc, g, result)
            continue
        end

        # build the per‐arm & combined dicts
        zdict = Dict{String,Union{Float64,Nothing}}()
        sigma = Dict{String,Union{Float64,Nothing}}()
        r2dict = Dict{String,Union{Float64,Nothing}}()

        if mode == "cis_only"
            zdict["cis"] = Zc
            zdict["combined"] = Zc
            sigma["cis"] = σ2c
            sigma["combined"] = σ2c
            r2dict["cis"] = r2c
            r2dict["combined"] = r2c

        elseif mode == "trans_only"
            zdict["trans"] = Zt
            zdict["combined"] = Zt
            sigma["trans"] = σ2t
            sigma["combined"] = σ2t
            r2dict["trans"] = r2t
            r2dict["combined"] = r2t

        else  # cis_trans
            zdict["cis"] = Zc
            zdict["trans"] = Zt
            zdict["combined"] = Zcomb
            sigma["cis"] = σ2c
            sigma["trans"] = σ2t
            sigma["combined"] = σ2comb
            r2dict["cis"] = r2c
            r2dict["trans"] = r2t
            r2dict["combined"] = r2comb
        end

        # p‐values
        p_dict = Dict{String,Union{Float64,Nothing}}()
        for (k, zv) in zdict
            p_dict[k] = zv === nothing ? nothing : Base.exp(logpvalue_z(zv))
        end

        # finally push
        push!(acc, g => Dict(
            "Z_score" => zdict,
            "pval" => p_dict,
            "sigma2" => sigma,
            "r2" => r2dict,
            "feature_size" => Dict("cis" => nc, "trans" => nt, "combined" => ncomb),
            "mode" => mode,
            "metrics" => get(result, "metrics", nothing),
            "feats_cis" => feats_c,
            "feats_trans" => feats_t,
            "weights" => get(get(result, "params", Dict()), "weights", nothing)
        ))
    end

    return tissue => Dict(acc)
end


# %%
# preallocate a spot for each result Pair
pairs = Vector{Pair{String,Dict{String,Any}}}(undef, length(tissues))

Threads.@threads for i in eachindex(tissues)
    t = tissues[i]
    pairs[i] = process_one_tissue(
        t,
        results_per_tissue,
        transposed_genotype_dict,
        exp_transposed,
        allele_alignment_dict
    )
end

# merge all into one Dict{String,Dict}
results_per_tissue_assoc = merge(Dict.(pairs)...)

# %%
results_per_tissue_assoc["AOR"]["gene6287"]

# %%
const MODES = ["cis", "trans", "combined"]

# %%
# using MultipleTesting: adjust, BenjaminiHochberg



for (tissue, gene_dict) in results_per_tissue_assoc
    # 1) collect all (gene,mode, logp) where logp ≠ nothing
    flat = Tuple{String,String,Float64}[]
    for (gene, vals) in gene_dict
        lmap = get(vals, "pval", nothing)  # now a Dict of log-p’s
        if lmap isa AbstractDict
            for m in MODES
                lp = get(lmap, m, nothing)
                lp !== nothing && push!(flat, (gene, m, lp))
            end
        end
    end

    # if nothing to adjust, skip
    isempty(flat) && continue

    # 2) exponentiate to get real p’s and BH–adjust
    real_ps = getindex.(flat, 3)

    adj_ps = adjust(real_ps, BenjaminiHochberg())

    # 3) build lookup (gene,mode) => log(adj_p)
    lookup = Dict{Tuple{String,String},Float64}()
    for (i, (g, m, _)) in enumerate(flat)
        lookup[(g, m)] = adj_ps[i]  # store in log scale
    end

    # 4) insert `pval_adj` (log‐scale) in place
    for (gene, vals) in gene_dict
        lmap = get(vals, "pval", nothing)
        ladj = Dict{String,Union{Float64,Nothing}}()
        if lmap isa AbstractDict
            for m in MODES
                if haskey(lmap, m) && lmap[m] !== nothing
                    ladj[m] = lookup[(gene, m)]
                else
                    ladj[m] = nothing
                end
            end
        end
        vals["pval_adj"] = ladj
    end
end







# %%
using JSON3, FilePathsBase  # for paths

# 1) A small sanitizer to turn any non-finite floats into JSON nulls
function sanitize(x)
    if x isa AbstractDict
        return Dict(k => sanitize(v) for (k, v) in x)
    elseif x isa AbstractVector
        return [sanitize(v) for v in x]
    elseif x isa Float64
        return isfinite(x) ? x : nothing
    else
        return x
    end
end

# 2) Loop over tissues and write each to its own file
outdir = joinpath(MODEL_DIR, "per_tissue_assoc_gp")
mkpath(outdir)

for (tissue, gene_dict) in results_per_tissue_assoc
    # sanitize this tissue’s dict
    clean = sanitize(gene_dict)

    # build a safe filename (e.g. replace spaces or slashes)
    fname = replace(tissue, r"[^\w]" => "_") * ".json"
    path = joinpath(outdir, fname)

    open(path, "w") do io
        JSON3.write(io, clean; pretty=true, indent=2)
    end
    @info "Wrote tissue $tissue to $path"
end


# %%
results_per_tissue_assoc["AOR"]

# %%
# pre-allocate
z_scores_per_tissue = Dict{String,Vector{Float64}}()

for (tissue, gene_dict) in results_per_tissue_assoc
    zs = Float64[]

    for v in values(gene_dict)
        # get the mode string, e.g. "cis_only", "trans_only" or "cis_trans"
        mode = get(v, "mode", "")

        # map the mode to the Z_score key
        key = mode == "cis_only" ? "cis" :
              mode == "trans_only" ? "trans" :
              mode == "cis_trans" ? "combined" :
              nothing

        if key !== nothing
            # pull the nested Z_score dict
            zmap = get(v, "Z_score", nothing)
            if zmap isa AbstractDict
                z = get(zmap, key, nothing)
                # only keep real, finite numbers
                # display(z)
                if z isa Number && isfinite(z)
                    push!(zs, Float64(z))
                end
            end
        end
    end

    z_scores_per_tissue[tissue] = zs
end


# %%
z_scores_per_tissue["SKLM"]

# %%
using StatsPlots

tissue_names = sort(collect(keys(z_scores_per_tissue)))
n_tissue = length(tissue_names)
ncols = 2#ceil(Int, sqrt(n_tissue))
nrows = ceil(Int, n_tissue / ncols)

plot_list = [StatsPlots.density(z_scores_per_tissue[t];
    title=t,
    # xlims=(-3, 3),
    xlabel="",
    ylabel="density",
    linewidth=2,
    margin=10mm,
    fill=(0, :lightblue),
    legend=false) for t in tissue_names]

plot_z = plot(plot_list...; layout=(nrows, ncols), size=(800, 1200), xlabel="TWAS Z-score")






# %%
PLOT_PATH = joinpath(DATA_DIR, "plots")
PLOT_ASSOC = joinpath(PLOT_PATH, "association")
z_score_path = joinpath(PLOT_ASSOC, "z_score_distribution.pdf")
gwas_z_score_path = joinpath(PLOT_ASSOC, "gwas_z_score_distribution.pdf")

# %%
savefig(plot_z, z_score_path)


# %%
using StatsPlots

tissue_names = sort(collect(keys(allele_alignment_dict)))
n_tissue = length(tissue_names)
ncols = 2
nrows = ceil(Int, n_tissue / ncols)

plot_list = [StatsPlots.density(allele_alignment_dict[t].Z;
    title=t,
    # xlims = (-3, 3),
    xlabel="",
    ylabel="",
    linewidth=2,
    margin=10mm,
    fill=(0, :lightblue),
    legend=false,
) for t in tissue_names]

plot_z = plot(plot_list...; layout=(nrows, ncols), size=(1000, 1200), xlabel="GWAS Z-score")


# %%


# %%
savefig(plot_z, gwas_z_score_path)



