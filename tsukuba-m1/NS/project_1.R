

e = c(1, 2, 1)
b = c(15, 1.75, 2)
qm = 100
# qm = -1.7587e+11
v = c(2, 3, 4)
# v = c(0, 1e5, 0)
x = c(10, 12, 1)
# x = c(0,0,0)

cross_product <- function(a, b) {
    r = rep(0, length(a))
    r[1] = a[2] * b[3] - a[3] * b[2]
    r[2] = -a[1] * b[3] + a[3] * b[1]
    r[3] = a[1] * b[2] - a[2] * b[1]
    return(r)
}


update_velocity <- function(part, e, b, delta_t) {
    t = qm * b * 0.5 * delta_t
    s = (2 * t) / (1 + t ^ 2)

    v_minus = part$v + qm * e * 0.5 * delta_t

    v_prime = v_minus + cross_product(v_minus, t)
    v_plus = v_minus + cross_product(v_prime, s)

    part$v = v_plus + qm * e * 0.5 * delta_t
    part$x <- part$x + part$v * delta_t
    return(part)
}

iterate <- function(part, it, delta_t) {
    pos <- part$x
    for (i in seq_len(it)) {
        part <- update_velocity(part, e, b, delta_t)
        pos <- rbind(pos, part$x)
    }
    return(pos)
}

iterate_new <- function(part, it, delta_t) {
    pos <- data.frame(X1 = part$x[1],
                      X2 = part$x[2],
                      X3 = part$x[3])
    for (i in seq_len(it)) {
        part <- update_velocity(part, e, b, delta_t)
        aux <-
            data.frame(X1 = part$x[1],
                       X2 = part$x[2],
                       X3 = part$x[3])
        pos <- plotly::rbindlist(list(pos, aux))
    }
    return(pos)
}



boris_algo <- function(it, e, b, qm, v, x, delta_t) {
    part = data.frame(x, v)
    pos <- iterate(part, it, delta_t)
    return(pos)

}

gerate_graph <- function(pos) {
  aux <- as.data.frame(pos)

  p <-
    plot_ly(
      aux,
      x = aux$V1,
      y = aux$V2,
      z = aux$V3,
      color = aux$X1,
      colors =  c('#4AC6B7', '#1972A4'),
      marker = list(size = 4, sizeref = 2)
    ) %>%
    add_markers() %>%
    layout(scene = list(
      xaxis = list(title = 'X'),
      yaxis = list(title = 'Y'),
      zaxis = list(title = 'Z')
    ))
  p
}


pos <- boris_algo(
    it = 1000,
    e = e,
    b = b,
    qm = qm,
    v = v,
    x = x,
    delta_t = 1e-5
)


print(gerate_graph(pos))
