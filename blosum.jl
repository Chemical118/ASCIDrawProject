using ProRF, Printf, JLD2, Statistics, DecisionTree, BioAlignments, XLSX, DataFrames, Random

@load "Save/seed.jld2" data_state learn_state imp_state

function get_amino_loc(R::AbstractRF, L::Vector{Tuple{Int, Char}})
    return [string(i[1] + R.amino_loc - 1) for i in L]
end

function blosum_matrix(num::Int)
    if num == 45
        return BLOSUM45
    elseif num == 50
        return BLOSUM50
    elseif num == 62
        return BLOSUM62
    elseif num == 80
        return BLOSUM80
    elseif num == 90
        return BLOSUM90
    else
        error(@sprintf "There are no Matrix such as BLOSUM%d" num)
    end
end

function parallel_predict_b(regr::RandomForestRegressor, L::Vector{Tuple{Int, Char}}, seq_vector::Vector{String}; blosum::Int=62)
    seq_vector = map(x -> x[map(y -> y[1], L)], seq_vector)

    blo = blosum_matrix(blosum)
    test_vector = Array{Vector{Float64}}(undef, length(seq_vector))
    Threads.@threads for i in eachindex(seq_vector)
        test_vector[i] = [Float64(blo[tar, s]) for ((_, tar), s) in zip(L, seq_vector[i])]
    end
    return DecisionTree.apply_forest(regr.ensemble, Matrix{Float64}(vcat(transpose.(test_vector)...)), use_multithreading=true)
end

function _find_key(dict::Dict{Char, Int}, tar::Int)
    for k in keys(dict)
        if dict[k] == tar
            return k
        end
    end
end

function get_data_b(R::AbstractRF, excel_col::Char; blosum::Int=62, norm::Bool=false, sheet::String="Sheet1", title::Bool=true)
    excel_data = DataFrames.DataFrame(XLSX.readtable(R.data_loc, sheet, infer_eltypes=title))
    excel_select_vector = excel_data[!, Int(excel_col) - Int('A') + 1]
    data_idx = findall(!ismissing, excel_select_vector)
    excel_select_vector = Vector{Float64}(excel_select_vector[data_idx])
    if norm == true
        excel_select_vector = min_max_norm(excel_select_vector)
    end

    data_len, loc_dict_vector, seq_matrix = ProRF._location_data(R.fasta_loc, data_idx)
    blo = blosum_matrix(blosum)
    x_col_vector = Vector{Vector{Float64}}()
    loc_vector = Vector{Tuple{Int, Char}}()
    for (ind, (dict, col)) in enumerate(zip(loc_dict_vector, eachcol(seq_matrix)))
        max_val = maximum(values(dict))
        max_amino = _find_key(dict, max_val)
        if '-' ∉ keys(dict) && 1 ≤ data_len - max_val 
            push!(x_col_vector, [blo[max_amino, i] for i in col])
            push!(loc_vector, (ind, max_amino))
        end
    end
    
    x = Matrix{Float64}(hcat(x_col_vector...))
    y = Vector{Float64}(excel_select_vector)
    l = Vector{Tuple{Int, Char}}(loc_vector)
    return x, y, l
end

function get_reg_importance_b(R::AbstractRF, X::Matrix{Float64}, Y::Vector{Float64}, L::Vector{Tuple{Int, Char}}, feat::Int, tree::Int;
    val_mode::Bool=false, test_size::Float64=0.3, memory_usage::Real=4.0, nbin::Int=200, show_number::Int=20, imp_iter::Int=60,
    max_depth::Int=-1, min_samples_leaf::Int=1, min_samples_split::Int=2,
    data_state::UInt64=@seed, learn_state::UInt64=@seed, imp_state::UInt64=@seed)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, data_state=data_state)
    regr = ProRF._randomforestregressor(feat, tree, max_depth, min_samples_leaf, min_samples_split, learn_state)
    DecisionTree.fit!(regr, x_train, y_train)

    if val_mode == false
        ProRF._view_result(regr, x_test, y_test, nbin)
    end
    memory_estimate = *(size(X)...) / 35000.0
    if memory_estimate > memory_usage
        n = size(X, 1)
        idx = shuffle(MersenneTwister(imp_state), 1:n)
        edit_idx = view(idx, 1:floor(Int, n * memory_usage / memory_estimate))
        X = X[edit_idx, :]
    end
    return regr, ProRF._rf_importance(regr, DataFrame(X, get_amino_loc(R, L)), imp_iter, seed=imp_state, show_number=show_number, val_mode=val_mode)
end

nothing # file end