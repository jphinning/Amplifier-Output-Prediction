# Amplifier Output Prediction

Modelagem comportamental de **amplificadores de potência (PA)** com Python, utilizando **Regressão Múltipla** e o **Modelo Polinomial com Memória (Memory Polynomial Model)** para predição da saída AM-AM e AM-PM do amplificador.

## 🎯 Objetivo

Predizer a saída não-linear de amplificadores de potência RF com efeitos de memória, utilizando modelos matemáticos clássicos de DPD (Digital Pre-Distortion) implementados em Python.

## 📐 Modelos Implementados

### Memory Polynomial Model (MPM)
O modelo polinomial com memória é amplamente utilizado em sistemas de comunicação para caracterização e linearização de PAs com não-linearidades dependentes de frequência:

$$y(n) = \sum_{k=1}^{K} \sum_{q=0}^{Q} a_{kq} \cdot x(n-q) \cdot |x(n-q)|^{k-1}$$

onde:
- `K` — ordem da não-linearidade
- `Q` — profundidade de memória
- `a_kq` — coeficientes estimados por regressão

### Regressão Múltipla (Least Squares)
Os coeficientes do modelo são estimados via mínimos quadrados (LS), minimizando o erro entre saída real e predita.

## 🛠️ Tecnologias

- **Python 3**
- **NumPy** — operações matriciais e álgebra linear
- **SciPy** — otimização e regressão
- **Matplotlib** — visualização AM-AM e AM-PM
- **Pandas** — manipulação de dados

## 📁 Estrutura

```
├── src/                        # Código fonte principal
│   ├── memory_polynomial.py    # Implementação do modelo MPM
│   ├── regression.py           # Estimação de coeficientes por LS
│   └── utils.py                # Funções auxiliares
├── requirements.txt            # Dependências Python
└── README.md
```

## 🚀 Como Executar

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar predição
python src/memory_polynomial.py
```

## 📊 Resultados

O modelo gera curvas de predição AM-AM (ganho) e AM-PM (distorção de fase) comparando a saída real do amplificador com a saída predita pelo modelo, avaliando o ajuste via métricas como NMSE (Normalized Mean Square Error).

## 📚 Referências

- Schetzen, M. — *The Volterra and Wiener Theories of Nonlinear Systems*
- Kenington, P. B. — *High Linearity RF Amplifier Design*
- Morgan, D. R. et al. — *A Generalized Memory Polynomial Model for Digital Predistortion of RF Power Amplifiers*, IEEE Trans. Signal Processing, 2006
