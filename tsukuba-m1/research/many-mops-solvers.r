suppressPackageStartupMessages(library(MOEADr))
suppressPackageStartupMessages(library(smoof))
suppressPackageStartupMessages(library(ecr))
suppressPackageStartupMessages(library(sensitivity))
suppressPackageStartupMessages(library(mlxR))
decomp <- list(name = "SLD")
decomp$H <- 16

aggfun <- list(name = "AWT")

neighbors <- list(name    = "x",
                  T       = 11,
                  delta.p = 0.909)

update <- list(name       = "restricted",
               UseArchive = TRUE)
update$nr = 1

scaling <- list(name = "simple")

constraint <- list(name = "none")

max.evals = round(20100/4)
stopcrit  <- list(list(name    = "maxeval",
                       maxeval = max.evals))

ZDT1 <- make_vectorized_smoof(prob.name  = "DTLZ2",
                                    dimensions = 60,
                                    n.objectives = 3)
problem.zdt1  <- list(name       = "ZDT1",
                      xmin       = rep(0, 60),
                      xmax       = rep(1, 60),
                      m          = 3)

variation <- list(list(name = "diffmut"),
                  list(name = "binrec"),
                  list(name = "binrec"),
                  list(name = "truncate"))
variation[[1]]$basis <- "rand"
variation[[1]]$Phi <- NULL
variation[[2]]$rho <- 0.495
variation[[3]]$rho <- 0.899

# summary(results.myvar2)
# plot(results.myvar2, suppress.pause = T)

results.orig2 <- MOEADr::moead(problem  = problem.zdt1,
                        preset   = preset_moead("original2"),
                        decomp = decomp,
                        stopcrit = stopcrit,
                        showpars = list(show.iters = "dots", showevery = 10),
                        seed     = 42)


# summary(results.orig2)
# plot(results.orig2, suppress.pause = T)

results.orig <- MOEADr::moead(problem  = problem.zdt1,
                       preset   = preset_moead("original"),
                       decomp = decomp,
                       stopcrit = stopcrit,
                       showpars = list(show.iters = "dots", showevery = 10),
                       seed     = 42)


# summary(results.orig)
# plot(results.orig, suppress.pause = T)

results.moead.de <- MOEADr::moead(problem  = problem.zdt1,
                      preset   = preset_moead("moead.de"),
                      decomp = decomp,
                      stopcrit = stopcrit,
                      showpars = list(show.iters = "dots", showevery = 10),
                      seed     = 42)


# summary(results.moead.de)
# plot(results.moead.de, suppress.pause = T)

fn <- makeDTLZ2Function(dimensions = 60, n.objectives = 3)

results.nsga2 = nsga2(
  fitness.fun = fn,
  n.objectives = 3,
  mu = 153,
  lambda = 153,
  lower = getLowerBoxConstraints(fn),
  upper = getUpperBoxConstraints(fn),
  terminators = list(stopOnEvals(max.evals)),
  minimize = T,
  n.dim = 60
)
computeHV(t(data.matrix(results.nsga2$pareto.front)))
# plot(results.nsga2$pareto.front)

results.smsemoa = smsemoa(
  fitness.fun = fn,
  n.objectives = 3,
  mu = 153,
  lower = getLowerBoxConstraints(fn),
  upper = getUpperBoxConstraints(fn),
  terminators = list(stopOnEvals(max.evals)),
  minimize = T,
  n.dim = 60
)

computeHV(t(data.matrix(results.smsemoa$pareto.front)))
plot(results.smsemoa$pareto.front)
# aspiration.set = matrix(
#   c(0.013, 0.021,
#     0, 0,
#     0.99, 0.885), ncol = 3L, byrow = FALSE
# )
# results.asemoa = asemoa(
#   fitness.fun = fn,
#   n.objectives = 2,
#   mu = 100,
#   lower = getLowerBoxConstraints(fn),
#   upper = getUpperBoxConstraints(fn),
#   terminators = list(stopOnEvals(max.evals)),
#   minimize = T,
#   n.dim = 60,
#   aspiration.set = aspiration.set
# )
# computeHV(t(data.matrix(results.asemoa$pareto.front)))
