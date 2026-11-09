# Demo — Clasificador sEMG

Demostración interactiva (Streamlit) del clasificador de gestos de mano para personas con
amputación transradial desarrollado en el trabajo de grado de Ingeniería Electrónica de la
Universidad del Valle (Cali, Colombia).

> El código de entrenamiento, los experimentos y el análisis completo están en el
> repositorio principal: **[jhonosorno-cmd/clasificador-semg](https://github.com/jhonosorno-cmd/clasificador-semg)**.
> Este repositorio contiene solo la aplicación de demostración.

---

## Ejecutar

```bash
python -m pip install -r requirements.txt
python -m streamlit run demo_app.py
```

`python -m streamlit` funciona aunque el comando `streamlit` no esté en el PATH
(frecuente en Windows con el Python de la Microsoft Store).

La demo lee `data/demo_cache.npz` (3,7 MB), que contiene el conjunto de prueba ya
preprocesado junto con las probabilidades de los cuatro modelos del ensemble.
**No requiere descargar NinaPro ni instalar PyTorch.** Si PyTorch está disponible
(`pip install torch`), habilita además la inferencia en vivo con los pesos de
`checkpoints/demo/`.

Permite explorar la matriz de confusión, el desempeño por clase, el efecto del umbral de
reposo sobre la exactitud global frente a la balanceada, la validación cruzada LORO y las
latencias medidas.

### Paquete portable (Windows, sin instalación)

```bash
python build_portable.py              # descarga dependencias con pip
python build_portable.py --no-torch   # sin PyTorch: usa el cache precomputado
```

Genera `dist/DemoEMG/`, una carpeta autocontenida con Python embebido que se ejecuta con
doble clic en `Iniciar_Demo.bat`, sin conexión ni permisos de administrador.

---

## Contenido

```
demo_app.py                          Aplicación Streamlit
build_portable.py                    Constructor del paquete portable para Windows
src/                                 Definición de los modelos (copiada del repositorio principal)
data/demo_cache.npz                  Conjunto de prueba precomputado (S1, repetición 6)
checkpoints/demo/                    Pesos de los cuatro modelos del ensemble
checkpoints/loro/loro_cv_summary.json  Resumen de la validación cruzada LORO
assets/fonts.css                     Fira Sans y Fira Code embebidas
.streamlit/config.toml               Tema y configuración sin telemetría
```

`data/demo_cache.npz` y los pesos se generan en el repositorio principal con
`python -m src.build_demo_cache`.

---

## Datos

Los datos provienen de **NinaPro DB3** (<https://ninapro.hevs.ch/>). El archivo
`data/demo_cache.npz` contiene 1 914 ventanas derivadas del sujeto 1, repetición 6
—filtradas, normalizadas y segmentadas— y no permite reconstruir el registro original.
Se publica bajo los términos de NinaPro, que exigen citar:

> Atzori, M., Gijsberts, A., Castellini, C., Caputo, B., Hager, A.-G. M., Elsig, S.,
> Giatsidis, G., Bassetto, F., & Müller, H. (2014). *Electromyography data for
> non-invasive naturally-controlled robotic hand prostheses.* Scientific Data, 1, 140053.
> https://doi.org/10.1038/sdata.2014.53

---

## Cita

Si usa esta demo o sus resultados, cite el trabajo de grado y el repositorio principal
(ver [CITATION.cff](CITATION.cff)):

> Osorno Pacheco, J. A. (2026). *Desarrollo de un sistema de clasificación de movimientos
> de la mano en personas con amputación de miembro superior usando aprendizaje profundo
> sobre señales mioeléctricas* [Trabajo de grado, Ingeniería Electrónica]. Universidad del
> Valle. Código: https://github.com/jhonosorno-cmd/clasificador-semg

---

## Licencia

Código bajo licencia [Apache 2.0](LICENSE); ver también [NOTICE](NOTICE). Los datos de
NinaPro DB3 no están cubiertos por esta licencia y se rigen por los términos de sus autores.
