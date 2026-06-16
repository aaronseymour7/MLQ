using Pkg
Pkg.activate("itensor_env")  
Pkg.instantiate()


using ITensors, ITensorMPS

function run(N::Int=150, J1::Float64=1.0, J2::Float64=0.0)
  sites = siteinds("S=1/2", N)

  weight = 20.0

  os = OpSum()
  for j = 1:N-1
    os += J1/2, "S+", j, "S-", j+1
    os += J1/2, "S-", j, "S+", j+1
    os += J1,   "Sz", j, "Sz", j+1
  end
  for j = 1:N-2
    os += J2/2, "S+", j, "S-", j+2
    os += J2/2, "S-", j, "S+", j+2
    os += J2,   "Sz", j, "Sz", j+2
  end

  H = MPO(os, sites)

  nsweeps = 30
  maxdim  = [10, 10, 10, 20, 20, 40, 80, 100, 200, 200]
  cutoff  = [1E-8]
  noise   = [1E-6]

  gs  = dmrg(H, random_mps(sites; linkdims=2); nsweeps, maxdim, cutoff, noise)
  ex1 = dmrg(H, [gs[2]],        random_mps(sites; linkdims=2); nsweeps, maxdim, cutoff, noise, weight)
  ex2 = dmrg(H, [gs[2], ex1[2]], random_mps(sites; linkdims=2); nsweeps, maxdim, cutoff, noise, weight)

  println("J2=$J2 | E0=$(gs[1])  E1=$(ex1[1])  E2=$(ex2[1])")
  println("       | gap(E1-E0)=$(ex1[1]-gs[1])  gap(E2-E0)=$(ex2[1]-gs[1])")
  println("       | <ψ1|ψ0>=$(inner(ex1[2],gs[2]))  <ψ2|ψ0>=$(inner(ex2[2],gs[2]))")
  println()
end

N = 200
for j2 in range(0.0, 1.0, length=21)
  run(N, 1.0, j2)
end
