# install.packages(c('tidyverse', 'caret', 'neuralnet', 'palmerpenguins'))
library(tidyverse)
library(caret)
library(neuralnet)
library(palmerpenguins)

# Se cargan los datos y se eliminan los registros con alguna celda vacía
datos <- penguins %>%
  na.omit() %>%
  select(species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g)

# Se normaliza de 0 a 1 para mayor precisión al entrenar, ya que al hacerlo
# con los valores reales, había un margen de error muy grande
normalizar <- function(x) (x - min(x)) / (max(x) - min(x))

# Guardamos los parámetros originales para poder desnormalizar si es necesario
params_norm <- datos %>%
  select(-species) %>%
  summarise(across(everything(), list(min = min, max = max)))

datos_norm <- datos %>%
  mutate(across(-species, normalizar))

# Verificar que todo quedó entre 0 y 1
summary(datos_norm)

# Se separa en train - test
muestra    <- createDataPartition(datos_norm$species, p = 0.8, list = F)
train_norm <- datos_norm[muestra, ]
test_norm  <- datos_norm[-muestra, ]


# Entrenamiento de la red neuronal
red.neuronal <- neuralnet(
  species ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g,
  data    = train_norm,
  hidden  = c(2, 3),
  stepmax = 1e+06
)

# Ver función de activación
red.neuronal$act.fct

# Visualizar la red
plot(red.neuronal)


# Realizar predicción sobre el test
prediccion <- predict(red.neuronal, test_norm, type = 'class')

specie.decod <- apply(prediccion, 1, which.max)
specie.pred  <- data_frame(specie.decod) %>%
  mutate(especie = recode(specie.decod,
                          "1" = "Adelie",
                          "2" = "Chinstrap",
                          "3" = "Gentoo"))

test_norm$species.pred <- specie.pred$especie

# Predicciones erróneas
erroneas <- test_norm[test_norm$species != test_norm$species.pred, ]
cat("Predicciones erróneas:", nrow(erroneas), "de", nrow(test_norm), "\n")
erroneas

# Exactitud
exactitud <- (1 - nrow(erroneas) / nrow(test_norm)) * 100
cat("Exactitud en test:", round(exactitud, 2), "%\n")

# Mostrar las matrices de pesos generadas
cat("W1 \n")
print(red.neuronal$weights[[1]][[1]])

cat("W2 \n")
print(red.neuronal$weights[[1]][[2]])

cat("W3 \n")
print(red.neuronal$weights[[1]][[3]])