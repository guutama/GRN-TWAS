# %%
#  GRN-TWAS modeling utilities 

# ── Packages ──────────────────────────────────────────────────────────────────
using Random, Statistics
using DataFrames
using JSON3
using Optim
using BioFindr
using Graphs
using ProgressMeter

using MLJ
using MLJLinearModels
using MLJScikitLearnInterface
using MLJBase: Table
using ScientificTypes: Continuous
using StatisticalMeasuresBase
using MLJEnsembles
using MLJModelInterface: DeterministicPipeline
using MLBase: Kfold

# Ensure MLJ pipelines accept continuous tables
import MLJModelInterface: input_scitype
input_scitype(::Type{<:DeterministicPipeline}) = Table{<:AbstractVector{<:Continuous}}

# ── Measures ──────────────────────────────────────────────────────────────────
struct ExplainedVariance end
function (m::ExplainedVariance)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    rss = sum((y .- ŷ).^2)
    tss = sum(y.^2)
    return 1 - rss / tss
end
StatisticalMeasuresBase.is_measure(::ExplainedVariance) = true
StatisticalMeasuresBase.orientation(::ExplainedVariance) = Score()
StatisticalMeasuresBase.observation_scitype(::ExplainedVariance) = Continuous
StatisticalMeasuresBase.external_aggregation_mode(::ExplainedVariance) = Mean()
StatisticalMeasuresBase.supports_weights(::ExplainedVariance) = false
StatisticalMeasuresBase.supports_class_weights(::ExplainedVariance) = false

# ── Model factories ───────────────────────────────────────────────────────────
const RidgeRegressorMLJ       = @load RidgeRegressor pkg=MLJLinearModels
const LinearRegressorMLJ      = @load LinearRegressor pkg=MLJLinearModels
const BayesianRidgeRegressor  = @load BayesianRidgeRegressor pkg=MLJScikitLearnInterface

function tune_model(X::DataFrame, y::Vector{<:Real},
                    X_val::DataFrame, y_val::Vector{<:Real},
                    rng; model_type::Symbol=:bayesian, k::Int=10)

    p = size(X, 2)
    if model_type == :bayesian
        base_model = BayesianRidgeRegressor(max_iter=10_000)
        ranges = [
            range(base_model, :alpha_1,  lower=1e-8, upper=100, scale=:log10),
            range(base_model, :alpha_2,  lower=1e-8, upper=100, scale=:log10),
            range(base_model, :lambda_1, lower=1e-8, upper=100, scale=:log10),
            range(base_model, :lambda_2, lower=1e-8, upper=100, scale=:log10),
            range(base_model, :tol,      lower=1e-6, upper=1e-3, scale=:log10),
        ]
    elseif model_type == :ridge
        base_model = RidgeRegressorMLJ()
        ranges = [range(base_model, :lambda, lower=1e-12, upper=1, scale=:log10)]
    elseif model_type == :linear
        base_model = LinearRegressorMLJ()
        ranges = []
    else
        error("Unknown model_type: choose :bayesian, :ridge, or :linear")
    end

    tuned_model = TunedModel(
        model=base_model,
        tuning=MLJParticleSwarmOptimization.ParticleSwarm(n_particles=10, rng=rng),
        resampling=CV(nfolds=k, shuffle=false, rng=rng),
        range=ranges,
        measure=[MLJ.rmse, ExplainedVariance()],
        n=30,
        acceleration=CPU1(),
    )

    pipe = Pipeline(standardizer=Standardizer(), regressor=tuned_model)
    mach = machine(pipe, X, y)
    fit!(mach, verbosity=0)

    y_pred  = MLJ.predict(mach, X_val)
    rmse    = MLJ.rms(y_pred, y_val)
    r2      = ExplainedVariance()(y_pred, y_val)
    rpt     = report(mach)
    cv_rmse = rpt.regressor.best_history_entry.measurement[1]
    cv_r2   = rpt.regressor.best_history_entry.measurement[2]

    return (mach=mach,
            metrics=(r2_val=r2, r2_cv=cv_r2, rmse_val=rmse, rmse_cv=cv_rmse),
            feature_size=p)
end

# ── Gene model (cis / trans / stacked) ────────────────────────────────────────
function gene_model(y::Vector{Float64},
                    X_cis::DataFrame,
                    X_trans::DataFrame;
                    y_val::Union{Nothing,Vector{Float64}}=nothing,
                    X_cis_val::Union{Nothing,DataFrame}=nothing,
                    X_trans_val::Union{Nothing,DataFrame}=nothing,
                    mode::Symbol,
                    seed::Int=42,
                    k::Int=10,
                    model_type::Symbol=:bayesian)

    rng = Random.MersenneTwister(seed)

    for df in (X_cis, X_trans, X_cis_val, X_trans_val)
        df === nothing && continue
        coerce!(df, autotype(df, :discrete_to_continuous))
    end

    if mode == :cis_only
        res = tune_model(X_cis, y, X_cis_val, y_val, rng; model_type, k)
        params = model_type == :bayesian ?
                 Dict(zip(names(X_cis), fitted_params(res.mach).regressor.best_fitted_params.coef)) :
                 fitted_params(res.mach).regressor.best_fitted_params
        return (metrics=res.metrics, models=res.mach, feature_size=res.feature_size,
                params=params, mode=:cis_only)

    elseif mode == :trans_only
        res = tune_model(X_trans, y, X_trans_val, y_val, rng; model_type, k)
        params = model_type == :bayesian ?
                 Dict(zip(names(X_trans), fitted_params(res.mach).regressor.best_fitted_params.coef)) :
                 fitted_params(res.mach).regressor.best_fitted_params
        return (metrics=res.metrics, models=res.mach, feature_size=res.feature_size,
                params=params, mode=:trans_only)

    elseif mode == :cis_trans
        shared   = intersect(names(X_cis), names(X_trans))
        Xc       = DataFrames.select(X_cis, Not(shared))
        Xc_val   = DataFrames.select(X_cis_val, Not(shared))
        Xt       = X_trans
        Xt_val   = X_trans_val

        if isempty(names(Xc))
            return gene_model(y, DataFrame(), Xt; y_val, X_cis_val=DataFrame(),
                              X_trans_val=Xt_val, mode=:trans_only, seed, k, model_type)
        elseif isempty(names(Xt))
            return gene_model(y, Xc, DataFrame(); y_val, X_cis_val=Xc_val,
                              X_trans_val=DataFrame(), mode=:cis_only, seed, k, model_type)
        end

        cis_res    = tune_model(Xc, y, Xc_val, y_val, rng; model_type, k)
        ŷcis_train = MLJ.unwrap(MLJ.predict(cis_res.mach, Xc))
        ŷcis_val   = MLJ.unwrap(MLJ.predict(cis_res.mach, Xc_val))

        resid_train = y .- ŷcis_train
        resid_val   = y_val .- ŷcis_val
        trans_res   = tune_model(Xt, resid_train, Xt_val, resid_val, rng; model_type, k)
        ŷtrans_val  = MLJ.unwrap(MLJ.predict(trans_res.mach, Xt_val))

        r2 = ExplainedVariance()
        obj(w) = -r2(w[1] .* ŷcis_val .+ w[2] .* ŷtrans_val, y_val)
        result = optimize(obj, [0.0, 0.0], [1.0, 1.0], [0.5, 0.5], Fminbox(BFGS()))
        w_cis, w_trans = Optim.minimizer(result)

        ŷ_final   = w_cis .* ŷcis_val .+ w_trans .* ŷtrans_val
        rmse_fin  = MLJ.rms(ŷ_final, y_val)
        r2_fin    = r2(ŷ_final, y_val)

        if model_type == :bayesian
            params_cis   = Dict(zip(names(Xc), fitted_params(cis_res.mach).regressor.best_fitted_params.coef))
            params_trans = Dict(zip(names(Xt), fitted_params(trans_res.mach).regressor.best_fitted_params.coef))
        else
            params_cis   = fitted_params(cis_res.mach).regressor.best_fitted_params
            params_trans = fitted_params(trans_res.mach).regressor.best_fitted_params
        end

        return (
            metrics=(cis=cis_res.metrics, trans=trans_res.metrics,
                     combined=(rmse_val=rmse_fin, r2_val=r2_fin)),
            models=(cis=cis_res.mach, trans=trans_res.mach),
            feature_size=(cis=cis_res.feature_size, trans=trans_res.feature_size,
                          stacked=cis_res.feature_size + trans_res.feature_size),
            params=(cis=params_cis, trans=params_trans, weights=(cis=w_cis, trans=w_trans)),
            mode=:cis_trans
        )
    else
        error("Unknown mode: $mode")
    end
end

# ── One-gene wrapper ──────────────────────────────────────────────────────────
function process_gene(gene_idx::Int,
                      expr::DataFrame, expr_val::DataFrame,
                      geno::DataFrame, geno_val::DataFrame,
                      cis_eqtls::Dict{String,Vector{String}},
                      trans_eqtls::Dict{String,Vector{String}},
                      idx2name::Dict{Int,String})

    gene_name = string(get(idx2name, gene_idx, "unknown"))
    @assert gene_name in names(expr) "Gene $gene_name is not in expression matrix"

    y     = Float64.(collect(expr[!, gene_name]))
    y_val = Float64.(collect(expr_val[!, gene_name]))

    has_cis   = haskey(cis_eqtls, gene_name)   && !isempty(cis_eqtls[gene_name])
    has_trans = haskey(trans_eqtls, gene_name) && !isempty(trans_eqtls[gene_name])

    cis_snps         = has_cis   ? intersect(cis_eqtls[gene_name],   names(geno)) : String[]
    trans_snps       = has_trans ? intersect(trans_eqtls[gene_name], names(geno)) : String[]
    valid_cis_snps   = intersect(unique(cis_snps), names(geno))
    valid_trans_snps = intersect(unique(trans_snps), names(geno))

    X_cis      = DataFrames.select(geno, valid_cis_snps)
    X_cis_val  = DataFrames.select(geno_val, valid_cis_snps)
    X_trans    = DataFrames.select(geno, valid_trans_snps)
    X_trans_val= DataFrames.select(geno_val, valid_trans_snps)

    valid_X_cis   = size(X_cis, 2)   > 0
    valid_X_trans = size(X_trans, 2) > 0

    mode = if     valid_X_cis && !valid_X_trans
        :cis_only
    elseif !valid_X_cis && valid_X_trans
        :trans_only
    elseif  valid_X_cis && valid_X_trans
        :cis_trans
    else
        return nothing
    end

    result = gene_model(y, X_cis, X_trans; y_val, X_cis_val, X_trans_val, mode)
    return gene_name => (
        models=result.models,
        metrices_and_params=(
            metrics=result.metrics,
            params=result.params,
            mode=result.mode,
            feature_size=result.feature_size
        )
    )
end



function main(dag_data::Tuple{Graphs.SimpleDiGraph,Dict{String,Int}},
                expr_tr::DataFrame,
                expr_val::DataFrame,
                geno_tr::DataFrame,
                geno_val::DataFrame,
                cis_eqtls::Dict{String,Vector{String}},
                trans_eqtls::Dict{String,Vector{String}},
                save_lock::ReentrantLock;
                checkpoint_interval::Int=100,
                gene_range::Union{AbstractVector{Int},String}="all",
                model_path::AbstractString=".")




# - dag_data = (G, name2idx)
#     G::Graphs.SimpleDiGraph over genes; name2idx maps gene_name::String → Int index in G.
#     Every gene name used in expression columns must appear in name2idx (and vice-versa).
#
# - expr_tr / expr_val (expression data,training/validation):
#     Rows   = samples; Columns = :genes
#     Size   = N_train × G and N_val × G, respectively.

#
# - geno_tr / geno_val (genotypes,training/validation   ):
#     Rows   = samples; Columns = :snp
#     Entries= 0/1/2 (Int);
#     The :sample column must match expr_tr/expr_val exactly and be in the same order.
#
# - cis_eqtls / trans_eqtls:
#     Dict{String,Vector{String}} mapping gene_name → vector of SNP ids (must match geno column names).
#

    if dag_data === missing || expr_tr === missing || geno_tr === missing || cis_eqtls === missing
        @warn "Skipping $tissue: missing data"
        return nothing
    end

    @assert expr_tr.sample == geno_tr.sample "Train sample order must match between expression and genotype"
    @assert expr_val.sample == geno_val.sample "Test sample order must match between expression and genotype"

    G, name2idx = dag_data
    idx2name = Dict(v => k for (k, v) in name2idx)

    sorted_genes = topological_sort(G)
    all_genes = gene_range == "all" ? sorted_genes : sorted_genes[gene_range]
    if isempty(all_genes)
        @warn "Skipping $tissue: no genes found in DAG"
        return tissue => Dict{String,Any}()
    end

    gene_results_list = Vector{Union{Nothing,Pair{String,Any}}}(undef, length(all_genes))
    @showprogress 1 "Processing $tissue genes..." for (i, gene_idx) in enumerate(all_genes)
        try
            gene_results_list[i] = process_gene(gene_idx, expr_tr, expr_val, geno_tr, geno_val,
                                                cis_eqtls, trans_eqtls, idx2name)
            if i % checkpoint_interval == 0
                filtered = filter(!isnothing, gene_results_list[1:i])
                out_dict = Dict(gene => payload[:metrices_and_params] for (gene, payload) in filtered)
                if !isempty(out_dict)
                    json_path = joinpath(model_path, "metrices_and_params_gp$(tissue).json")
                    open(json_path, "w") do io
                        JSON3.write(io, out_dict; indent=2)
                    end
                end
            end
        catch e
            @warn "Gene $(idx2name[gene_idx]) in $tissue failed: $e"
            gene_results_list[i] = nothing
        end
    end

    tissue_pairs = filter(!isnothing, gene_results_list)
    gene_dict = Dict(tissue_pairs)
    return  gene_dict
end




# Before running the main function, ensure you have the required inputs:
# ── Run Main  ────────────────────────────
result_tissue = main(dag_data::Tuple{Graphs.SimpleDiGraph,Dict{String,Int}},
                expr_tr::DataFrame,
                expr_val::DataFrame,
                geno_tr::DataFrame,
                geno_val::DataFrame,
                cis_eqtls::Dict{String,Vector{String}},
                trans_eqtls::Dict{String,Vector{String}},
                save_lock::ReentrantLock;
                checkpoint_interval::Int=100,
                gene_range::Union{AbstractVector{Int},String}="all",
                model_path::AbstractString=".")

  

    gene_dict = result_tissue
    out_dict = Dict{String,Any}()
    for (gene, payload) in gene_dict
        out_dict[gene] = payload[:metrices_and_params]
    end

    json_path = joinpath(model_path, "metrices_and_params.json")
    mkpath(dirname(json_path))
    open(json_path, "w") do io
        JSON3.write(io, out_dict; indent=2)
    end
    

    @info "Saved $(length(out_dict)) genes to $json_path"
 

    
#=
    julia --project -t auto run_main.jl \
  --dag data/AOR_dag.jld2 \
  --expr-tr data/AOR_expr_train.csv \
  --expr-val data/AOR_expr_val.csv \
  --geno-tr data/AOR_geno_train.csv \
  --geno-val data/AOR_geno_val.csv \
  --cis data/AOR_cis_eqtls.json \
  --trans data/AOR_trans_eqtls.json \
  --model-path outputs/AOR \
  --checkpoint-interval 200 \
  --gene-range "all"
  =#