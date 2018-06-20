suppressPackageStartupMessages(library(ecr))
library(R.utils)
setwd("~/Documents/estudos/MOEADr/R/")
sourceDirectory(".")
file.sources = list.files(pattern = "*.R")

library(plotly)
decomp <- list(name = "SLD")
decomp$H <- 16

max.evals = round(20100*3)
stopcrit  <- list(list(name    = "maxeval",
                       maxeval = max.evals))

ZDT1 <- MOEADr::make_vectorized_smoof(prob.name  = "DTLZ1",
                                      dimensions = 10,
                                      n.objectives = 2)
problem.zdt1  <- list(
  name       = "ZDT1",
  xmin       = rep(0, 10),
  xmax       = rep(1, 10),
  m          = 2
)

problem <- problem.zdt1

set.seed(42)
# control <- MOEADr::moead(problem  = problem.zdt1,
#                                   preset   = preset_moead("moead.de"),
#                                   decomp = decomp,
#                                   stopcrit = stopcrit,
#                                   showpars = list(show.iters = "dots", showevery = 500),
#                                   seed     = 43)

increments <- list(lhs = NULL,
                   ONRA = list(onra = NULL, dt = 20))


hv.moead.de <- list()
hv.moead.de.ORA <- list()
hv.moead.de.ORA.lhs <- list()
hv.moead.de.lhs <- list()
hv.moead.de.cga <- list()
hv.moead.de.cga.ORA <- list()
hv.moead.de.cga.lhs <- list()
hv.moead.de.cga.ORA.lhs <- list()

control1 <- MOEADr::moead(
  problem  = problem.zdt1,
  preset   = preset_moead("moead.de"),
  decomp = decomp,
  stopcrit = stopcrit,
  showpars = list(show.iters = "dots", showevery = 50),
  seed     = i
)

for (i in 1:10) {
  print(i)
  print("results.moead.de")
  results.moead.de <- moead(
    problem  = problem.zdt1,
    preset   = preset_moead("moead.de"),
    decomp = decomp,
    stopcrit = stopcrit,
    showpars = list(show.iters = "dots", showevery = 50),
    seed     = i
  )
  hv.moead.de[[length(hv.moead.de) + 1]] <-
    emoa::dominated_hypervolume(points = t(apply(
      results.moead.de$Y, MARGIN = 2, scale_vector
    )), ref = ref.points)# higher is better


  increments <- list(lhs = NULL,
                     ONRA = list(onra = TRUE, dt = 20))
  print("results.moead.de.ORA")
  results.moead.de.ORA <- moead(
    problem  = problem.zdt1,
    preset   = preset_moead("moead.de"),
    decomp = decomp,
    stopcrit = stopcrit,
    increments = increments,
    showpars = list(show.iters = "dots", showevery = 50),
    seed     = i
  )
  hv.moead.de.ORA[[length(hv.moead.de.ORA) + 1]] <-
    emoa::dominated_hypervolume(points = t(apply(
      results.moead.de.ORA$Y, MARGIN = 2, scale_vector
    )), ref = ref.points)# higher is better

  increments <- list(lhs = TRUE,
                     ONRA = list(onra = TRUE, dt = 20))

  print("results.moead.de.ORA.lhs")
  results.moead.de.ORA.lhs <- moead(
    problem  = problem.zdt1,
    preset   = preset_moead("moead.de"),
    decomp = decomp,
    stopcrit = stopcrit,
    increments = increments,
    showpars = list(show.iters = "dots", showevery = 50),
    seed     = i
  )
  hv.moead.de.ORA.lhs[[length(hv.moead.de.ORA.lhs) + 1]] <-
    emoa::dominated_hypervolume(points = t(apply(
      results.moead.de.ORA.lhs$Y, MARGIN = 2, scale_vector
    )), ref = ref.points)# higher is better


  increments <- list(lhs = TRUE,
                     ONRA = list(onra = NULL, dt = 20))

  print("results.moead.de.lhs")
  results.moead.de.lhs <- moead(
    problem  = problem.zdt1,
    preset   = preset_moead("moead.de"),
    decomp = decomp,
    stopcrit = stopcrit,
    increments = increments,
    showpars = list(show.iters = "dots", showevery = 50),
    seed     = i
  )
  hv.moead.de.lhs[[length(hv.moead.de.lhs) + 1]] <-
    emoa::dominated_hypervolume(points = t(apply(
      results.moead.de.lhs$Y, MARGIN = 2, scale_vector
    )), ref = ref.points)# higher is better

  neighbors <- list(
    name    = "cga",
    T       = 4,
    LR       = 3,
    delta.p = 0.909
  )


  print("results.moead.de.cga")
  results.moead.de.cga <- moead(
    problem  = problem.zdt1,
    preset   = preset_moead("moead.de"),
    neighbors = neighbors,
    decomp = decomp,
    stopcrit = stopcrit,
    showpars = list(show.iters = "dots", showevery = 50),
    seed     = i
  )
  hv.moead.de.cga[[length(hv.moead.de.cga) + 1]] <-
    emoa::dominated_hypervolume(points = t(apply(
      results.moead.de.cga$Y, MARGIN = 2, scale_vector
    )), ref = ref.points)# higher is better

  neighbors <- list(
    name    = "cga",
    T       = 4,
    LR       = 3,
    delta.p = 0.909
  )

  increments <- list(lhs = NULL,
                     ONRA = list(onra = TRUE, dt = 20))

  print("results.moead.de.cga.ORA")
  results.moead.de.cga.ORA <- moead(
    problem  = problem.zdt1,
    preset   = preset_moead("moead.de"),
    neighbors = neighbors,
    decomp = decomp,
    stopcrit = stopcrit,
    increments = increments,
    showpars = list(show.iters = "dots", showevery = 50),
    seed     = i
  )
  hv.moead.de.cga.ORA[[length(hv.moead.de.cga.ORA) + 1]] <-
    emoa::dominated_hypervolume(points = t(apply(
      results.moead.de.cga.ORA$Y, MARGIN = 2, scale_vector
    )), ref = ref.points)# higher is better


  neighbors <- list(
    name    = "cga",
    T       = 4,
    LR       = 3,
    delta.p = 0.909
  )


  increments <- list(lhs = TRUE,
                     ONRA = list(onra = NULL, dt = 20))

  print("results.moead.de.cga.lhs")
  results.moead.de.cga.lhs <- moead(
    problem  = problem.zdt1,
    preset   = preset_moead("moead.de"),
    neighbors = neighbors,
    decomp = decomp,
    stopcrit = stopcrit,
    increments = increments,
    showpars = list(show.iters = "dots", showevery = 50),
    seed     = i
  )
  hv.moead.de.cga.lhs[[length(hv.moead.de.cga.lhs) + 1]] <-
    emoa::dominated_hypervolume(points = t(apply(
      results.moead.de.cga.lhs$Y, MARGIN = 2, scale_vector
    )), ref = ref.points)# higher is better

  neighbors <- list(
    name    = "cga",
    T       = 4,
    LR       = 3,
    delta.p = 0.909
  )


  increments <- list(lhs = TRUE,
                     ONRA = list(onra = TRUE, dt = 20))
  print("results.moead.de.cga.ORA.lhs")
  results.moead.de.cga.ORA.lhs <- moead(
    problem  = problem.zdt1,
    preset   = preset_moead("moead.de"),
    neighbors = neighbors,
    decomp = decomp,
    stopcrit = stopcrit,
    increments = increments,
    showpars = list(show.iters = "dots", showevery = 50),
    seed     = i
  )
  hv.moead.de.cga.ORA.lhs[[length(hv.moead.de.cga.ORA.lhs) + 1]] <-
    emoa::dominated_hypervolume(points = t(
      apply(results.moead.de.cga.ORA.lhs$Y, MARGIN = 2, scale_vector)
    ), ref = ref.points)# higher is better

}

generate_onraph <- function(res) {
  aux <- data.frame(res$Y)

  p <-
    plot_ly(
      aux,
      x = aux$f1,
      y = aux$f2,
      z = aux$f3,
      color = aux$X1,
      colors =  c('#4AC6B7', '#1972A4')
    ) %>%
    add_markers() %>%
    layout(scene = list(
      xaxis = list(title = 'X'),
      yaxis = list(title = 'Y'),
      zaxis = list(title = 'Z')
    ))
  p
}
generate_onraph_nsga <- function(res) {
  aux <- data.frame(res$pareto.front)

  p <-
    plot_ly(
      aux,
      x = aux$y1,
      y = aux$y2,
      z = aux$y3,
      color = aux$X1,
      colors =  c('#4AC6B7', '#1972A4')
    ) %>%
    add_markers() %>%
    layout(scene = list(
      xaxis = list(title = 'X'),
      yaxis = list(title = 'Y'),
      zaxis = list(title = 'Z')
    ))
  p
}


df.hv.moead.de <- as.data.frame(cbind(unlist(hv.moead.de), 1))
df.hv.moead.de.ORA <-
  as.data.frame(cbind(unlist(hv.moead.de.ORA), 2))
df.hv.moead.de.ORA.lhs <-
  as.data.frame(cbind(unlist(hv.moead.de.ORA.lhs), 3))
df.hv.moead.de.lhs <-
  as.data.frame(cbind(unlist(hv.moead.de.lhs), 4))
df.hv.moead.de.cga <-
  as.data.frame(cbind(unlist(hv.moead.de.cga), 5))
df.hv.moead.de.cga.ORA <-
  as.data.frame(cbind(unlist(hv.moead.de.cga.ORA), 6))
df.hv.moead.de.cga.lhs <-
  as.data.frame(cbind(unlist(hv.moead.de.cga.lhs), 7))
df.hv.moead.de.cga.ORA.lhs <-
  as.data.frame(cbind(unlist(hv.moead.de.cga.ORA.lhs), 8))

df.anova <-
  rbind(
    df.hv.moead.de,
    df.hv.moead.de.ORA,
    df.hv.moead.de.ORA.lhs,
    df.hv.moead.de.lhs,
    df.hv.moead.de.cga,
    df.hv.moead.de.cga.ORA,
    df.hv.moead.de.cga.lhs,
    df.hv.moead.de.cga.ORA.lhs
  )
boxplot(df.anova$V1~df.anova$V2)
# HV -> convergency, uniformity, spread
# IGD -> convergency, uniformity, spread
# R2 -> convergency
# epsilon_indicator -> convergency
# ref.points <- c(3.367973, 2.456659, 2.882586)
ref.points <- rep(round(1 + 1 / decomp$H, 3), problem.zdt1$m)
ideal <- rep(0, problem.zdt1$m)

summary(control)
summary(results.moead.de)
emoa::dominated_hypervolume(points = t(apply(
  results.moead.de$Y, MARGIN = 2, scale_vector
)), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.moead.de$Y), o = matrix(ref.points))# smaller is better
emoa::unary_r2_indicator(results.moead.de$Y,
                         matrix(results.moead.de$W),
                         matrix(ideal))# smaller is better
MOEADr::calcIGD(scales::rescale(t(results.moead.de$Y)), matrix(ref.points))# higher is better
# plot(results.moead.de, suppress.pause = T)
# print(generate_onraph(results.moead.de))



summary(results.moead.de.ORA)
emoa::dominated_hypervolume(points = t(apply(
  results.moead.de.ORA$Y, MARGIN = 2, scale_vector
)), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.moead.de.ORA$Y), o = matrix(ref.points))# smaller is better
emoa::unary_r2_indicator(results.moead.de.ORA$Y,
                         matrix(results.moead.de$W),
                         matrix(ideal))# smaller is better
MOEADr::calcIGD(scales::rescale(t(results.moead.de.ORA$Y)), matrix(ref.points))# higher is better
# plot(results.moead.de.ORA, suppress.pause = T)
# print(generate_onraph(results.moead.de.ORA))



summary(results.moead.de.ORA.lhs)
emoa::dominated_hypervolume(points = t(apply(
  results.moead.de.ORA.lhs$Y, MARGIN = 2, scale_vector
)),
ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.moead.de.ORA.lhs$Y), o = matrix(ref.points))# smaller is better
emoa::unary_r2_indicator(results.moead.de.ORA.lhs$Y,
                         matrix(results.moead.de$W),
                         matrix(ideal))# smaller is better
MOEADr::calcIGD(scales::rescale(t(results.moead.de.ORA.lhs$Y)), matrix(ref.points))# higher is better
# plot(results.moead.de.ORA.lhs, suppress.pause = T)
# print(generate_onraph(results.moead.de.ORA.lhs))


summary(results.moead.de.lhs)
emoa::dominated_hypervolume(points = t(apply(
  results.moead.de.lhs$Y, MARGIN = 2, scale_vector
)), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.moead.de.lhs$Y), o = matrix(ref.points))# smaller is better
emoa::unary_r2_indicator(results.moead.de.lhs$Y,
                         matrix(results.moead.de$W),
                         matrix(ideal))# smaller is better
MOEADr::calcIGD(scales::rescale(t(results.moead.de.lhs$Y)), matrix(ref.points))# higher is better
# plot(results.moead.de.lhs, suppress.pause = T)
# print(generate_onraph(results.moead.de.lhs))


summary(results.moead.de.cga)
emoa::dominated_hypervolume(points = t(apply(
  results.moead.de.cga$Y, MARGIN = 2, scale_vector
)), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.moead.de.cga$Y), o = matrix(ref.points))# smaller is better
emoa::unary_r2_indicator(
  results.moead.de.cga$Y,
  matrix(results.moead.de$W),
  matrix(results.moead.de$nadir)
)# smaller is better
MOEADr::calcIGD(scales::rescale(t(results.moead.de.cga$Y)), matrix(ref.points))# higher is better
# plot(results.moead.de.cga, suppress.pause = T)
# print(generate_onraph(results.moead.de.cga))


summary(results.moead.de.cga.ORA)
emoa::dominated_hypervolume(points = t(apply(
  results.moead.de.cga.ORA$Y, MARGIN = 2, scale_vector
)),
ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.moead.de.cga.ORA$Y), o = matrix(ref.points))# smaller is better
emoa::unary_r2_indicator(
  results.moead.de.cga.ORA$Y,
  matrix(results.moead.de$W),
  matrix(results.moead.de$nadir)
)# smaller is better
MOEADr::calcIGD(scales::rescale(t(results.moead.de.cga.ORA$Y)), matrix(ref.points))# higher is better
# plot(results.moead.de.cga.ORA, suppress.pause = T)
# print(generate_onraph(results.moead.de.cga.ORA))


summary(results.moead.de.cga.ORA.lhs)
emoa::dominated_hypervolume(points = t(
  apply(results.moead.de.cga.ORA.lhs$Y, MARGIN = 2, scale_vector)
),
ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.moead.de.cga.ORA.lhs$Y), o = matrix(ref.points))# smaller is better
emoa::unary_r2_indicator(
  results.moead.de.cga.ORA.lhs$Y,
  matrix(results.moead.de$W),
  matrix(results.moead.de$nadir)
)# smaller is better
MOEADr::calcIGD(scales::rescale(t(results.moead.de.cga.ORA.lhs$Y)), matrix(ref.points))# higher is better
# plot(results.moead.de.cga.ORA.lhs, suppress.pause = T)
# print(generate_onraph(results.moead.de.cga.ORA.lhs))

summary(results.moead.de.cga.lhs) # best
emoa::dominated_hypervolume(points = t(apply(
  results.moead.de.cga.lhs$Y, MARGIN = 2, scale_vector
)), ref = ref.points)# higher is better
emoa::epsilon_indicator(t(results.moead.de.cga.lhs$Y), o =  matrix(ref.points))# smaller is better
emoa::unary_r2_indicator(
  results.moead.de.cga.lhs$Y,
  matrix(results.moead.de$W),
  matrix(results.moead.de$nadir)
)# smaller is better
MOEADr::calcIGD(scales::rescale(t(results.moead.de.cga.lhs$Y)), matrix(ref.points))# higher is better
# plot(results.moead.de.cga.lhs, suppress.pause = T)
# print(generate_onraph(results.moead.de.cga.lhs))

# print(generate_onraph(results.moead.de))
# print(generate_onraph(results.moead.de.ORA))
# print(generate_onraph(results.moead.de.ORA.lhs))
# print(generate_onraph(results.moead.de.lhs))
# print(generate_onraph(results.moead.de.cga))
# print(generate_onraph(results.moead.de.cga.ORA))
# print(generate_onraph(results.moead.de.cga.ORA.lhs))
# print(generate_onraph(results.moead.de.cga.lhs))
