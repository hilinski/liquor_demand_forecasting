# 📊 Liquor Demand Forecasting

Este repositorio contiene el proyecto final del **Bootcamp de Ciencia de Datos de Le Wagon Batch 1767 (2024-2025)**. 

## 📌 Problema

Las empresas necesitan predecir su demanda para poder coordinar su operación y definir acciones comerciales estratégicas. Sin embargo, proyectar la demanda de productos de consumo masivo, como los licores, presenta desafíos como:

- Muchas categorías de productos.
- Gran cantidad de clientes y transacciones.
- Alta competencia en el mercado.
- Falta de metodologías basadas en datos en la toma de decisiones.

Además, la falta de herramientas adecuadas eleva el costo de generar proyecciones precisas, lo que afecta la planificación y rentabilidad de las empresas.

## 📊 Dataset Utilizado

Utilizamos el dataset público de BigQuery: **`bigquery-public-data.iowa_liquor_sales.sales`**, el cual contiene información detallada sobre ventas de licores en Iowa.

Ejemplo de la tabla:

```sql
SELECT * 
FROM `bigquery-public-data.iowa_liquor_sales.sales` 
LIMIT 5;
```

| invoice_and_item_number | date       | store_number | store_name                  | city       | category_name          | vendor_name                | item_description                | bottles_sold | sale_dollars |
|-------------------------|------------|--------------|-----------------------------|------------|------------------------|----------------------------|--------------------------------|--------------|--------------|
| RINV-05571600015       | 2024-12-23 | 2515         | HY-VEE FOOD STORE #1        | MASON CITY | AMERICAN VODKAS        | CONSTELLATION BRANDS INC   | SVEDKA 80PRF                    | -60          | -1206.0      |
| RINV-05397800006       | 2024-08-21 | 2548         | HY-VEE FOOD STORE           | ALTOONA    | IMPORTED VODKAS        | SAZERAC COMPANY INC        | FRIS DANISH VODKA               | -54          | -687.96      |
| RINV-04915400095       | 2023-10-23 | 5916         | ANOTHER ROUND               | DEWITT     | TENNESSEE WHISKIES     | OLE SMOKY DISTILLERY LLC   | OLE SMOKY COOKIE DOUGH WHISKEY | -30          | -495.0       |
| RINV-04399300115       | 2022-11-30 | 2670         | HY-VEE FOOD STORE           | CORALVILLE | 100% AGAVE TEQUILA     | BROWN FORMAN CORP.         | EL JIMADOR SILVER               | -24          | -462.0       |
| RINV-05247000010       | 2024-05-06 | 2515         | HY-VEE FOOD STORE #1        | MASON CITY | CANADIAN WHISKIES      | HEAVEN HILL BRANDS         | BLACK VELVET                     | -6           | -103.5       |

## 🧪 Modelos Implementados

Inicialmente, intentamos aplicar **Redes Neuronales Recurrentes (RNN)** para modelar la serie temporal de ventas. Sin embargo, enfrentamos las siguientes dificultades:

- **Generación de secuencias:** Había demasiadas combinaciones posibles entre condados y categorías de licores.
- **Codificación de variables categóricas:** Al usar **one-hot encoding**, el número de columnas aumentó drásticamente.
- **Interpretación de picos de demanda:** No existían variables explicativas claras para eventos como festividades o descuentos.

### 🔄 Cambio de Estrategia: SARIMA

Dado que el enfoque con RNN no produjo resultados satisfactorios, optamos por **simplificar el problema** agrupando las ventas por **semana y categoría de licor**. 

- **Ejemplo:** Se agruparon **Tequila** y **Mezcal** bajo una sola categoría.
- **Modelo elegido:** **SARIMA (Seasonal AutoRegressive Integrated Moving Average)**
- **Motivo:** Maneja estacionalidad y permite un análisis más interpretable de las tendencias.

### 📈 Comparación de Resultados

| Modelo | Error MAE | Interpretabilidad |
|--------|----------|------------------|
| RNN    | Alto     | Baja             |
| SARIMA | Bajo     | Alta             |

Los resultados fueron significativamente mejores con SARIMA, con predicciones más ajustadas a los patrones observados en los datos históricos.

## 🚀 Implementación

Este proyecto cuenta con una **interfaz gráfica** desarrollada con **FastAPI y Streamlit**, donde se pueden visualizar las predicciones.

🔗 [Repositorio del Frontend](https://github.com/hilinski/liquor_demand_forecasting_streamlit)

## 👨‍💻 Equipo

- **Renan Gameiro** - [GitHub](https://github.com/hilinski)
- **Anderson Andrade** - [GitHub](https://github.com/alphanetEX)
- **David Alatorre** - [GitHub](https://github.com/davidSA10)
- **Felipe Escobar** - [GitHub](https://github.com/pipesco93)

---

