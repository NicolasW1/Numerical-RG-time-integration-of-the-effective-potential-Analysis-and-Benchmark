parentDirectory(path, n=1) = joinpath(splitpath(normpath(path))[1:end-n])

function get_sparsity_pattern(u0, kernel!::Function, p)
    du0 = copy(u0)
    return Symbolics.jacobian_sparsity((du, u) -> kernel!(du, u, p, 0.0), du0, u0)
end

# takes a list of tuples, e.g. [(0.1, 20), (0.3, 15)]
# produces a grid with sum(n_i) - (n_Tuple-1) points
# if it has only one element you can also use generateGrid(ρmax, n)
function generateGrid(inputList)
    T = typeof(inputList[1][1])

    nt = sort(inputList, by=x -> x[1])
    grid = zeros(T, 0)
    prevρ = zero(T)
    for (ρmax, n) in nt
        append!(grid, collect(range(prevρ, ρmax; length=n)))
        prevρ = ρmax
    end

    unique(grid)
end
generateGrid(tuple::Tuple) = generateGrid([tuple])
generateGrid(ρmax, n) = generateGrid([(ρmax, n)])