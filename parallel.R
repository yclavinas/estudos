library(parallel)

library(foreach)
library(doParallel)

decomp <- list(name = "SLD")
decomp$H <- 16

max.evals = round(20100/4)
stopcrit  <- list(list(name    = "maxeval",
                       maxeval = max.evals))

ZDT1 <- MOEADr::make_vectorized_smoof(prob.name  = "DTLZ1",
                                      dimensions = 60,
                                      n.objectives = 2)
problem.zdt1  <- list(
  name       = "ZDT1",
  xmin       = rep(0, 60),
  xmax       = rep(1, 60),
  m          = 2
)

problem <- problem.zdt1


cl <- makeCluster(3)
registerDoParallel(cl)
clusterEvalQ(cl, library(MOEADr))
clusterExport(cl, "problem.zdt1")
clusterExport(cl, "stopcrit")
clusterExport(cl, "generate_weights")
clusterExport(cl, "decomp")
clusterExport(cl, "create_population")
clusterExport(cl, "evaluate_population")
clusterExport(cl, "denormalize_population")
clusterExport(cl, "decomposition_sld")
clusterExport(cl, "ZDT1")
clusterExport(cl, "define_neighborhood")
clusterExport(cl, "is_within")
clusterExport(cl, "perform_variation")
clusterExport(cl, "variation_sbx")
clusterExport(cl, "randM")
clusterExport(cl, "calc_Betaq")
clusterExport(cl, "variation_polymut")
clusterExport(cl, "calc_Deltaq")
clusterExport(cl, "neighborhood_cga")
clusterExport(cl, "control")
clusterExport(cl, "increments")
clusterExport(cl, "getminP")
clusterExport(cl, "getmaxP")

parWrapper <- function(seed) {
  neighbors <- list(
    name    = "cga",
    T       = 5,
    LR       = 8,
    delta.p = 0.909
  )
  instance <- moead(
      problem  = problem.zdt1,
      preset   = preset_moead("moead.de"),
      decomp = decomp,
      neighbors = neighbors,
      showpars = list(show.iters = "dots", showevery = 100),
      seed = seed
    )
  ref.points <- rep(1 + 1 / instance$H, instance$n.problems)
  print(summary(instance))
  hv <-
    emoa::dominated_hypervolume(points = t(apply(instance$Y, MARGIN = 2, scale_vector)), ref = ref.points)# higher is better
  output <- list(
    hv = hv,
    seed = seed
  )
}

a <-
  foreach(seed = 1:3, .combine = rbind)  %dopar%  parWrapper()
print(a)

stopImplicitCluster()
