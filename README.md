# Projeto05

## Algoritmo Naive Bayes - Dados de crédito alemão

**Descrição:** O presente projeto tem como objetivo a utilização do algoritmo Naive Bayes para solucionar um problema de classificação.

<div align='justify'>

**Contextualização:** Quando alguém pede um empréstimo ao banco, a instituição financeira precisa avaliar se esse empréstimo é um bom negócio ou não. Nessa avaliação, dois riscos estão em jogo: se a pessoa tem um histórico bom de crédito e é capaz de pagar o empréstimo, o banco perde a oportunidade de fazer negócio ao não aprovar o pedido. Mas se a pessoa tem um histórico ruim de crédito e não é confiável para pagar o empréstimo, o banco corre o risco de perder dinheiro ao aprovar o empréstimo.

Para reduzir os prejuízos, o banco precisa de uma regra que oriente quem deve receber a aprovação do empréstimo e quem não deve. Nesse processo, os gestores de empréstimos levam em consideração diversos fatores, como informações demográficas e socioeconômicas do solicitante, antes de tomar uma decisão sobre o pedido de empréstimo.

**Objetivo da Análise:** Treinar um modelo de Machine Learning usando o algoritmo Naive Bayes para prever se novos solicitantes de empréstimo serão bons ou maus pagadores.

**Conjunto de dados:** Os dados são sobre 1000 clientes de um banco alemão que fizeram solicitação de empréstimo e tem uma classe que diz se eles foram bons ou maus pagadores. 

### Sobre o Naive Bayes

O algoritmo de machine learning Naive Bayes é um método probabilístico baseado no Teorema de Bayes, que é utilizado para classificação de dados. A ideia é encontrar a probabilidade de uma instância pertencer a uma classe, com base nas suas características.

O algoritmo parte do princípio de independência entre as características (daí o nome "naive" ou "ingênuo"), o que significa que a presença ou ausência de uma característica não afeta a presença ou ausência de outra característica. Essa suposição simplifica os cálculos necessários para determinar as probabilidades de uma instância pertencer a uma classe.

O modelo é treinado com um conjunto de dados rotulados e, a partir disso, ele calcula as probabilidades de cada classe, bem como as probabilidades de cada característica para cada classe. Na fase de teste, o algoritmo calcula a probabilidade de uma nova instância pertencer a cada classe, utilizando as probabilidades previamente calculadas. Em seguida, ele classifica a instância na classe com a maior probabilidade.

O Naive Bayes é frequentemente utilizado em problemas de classificação com múltiplas classes, tais como classificação de textos e detecção de spam. Ele é simples, rápido e fácil de implementar, além de exigir poucos recursos computacionais.

### Sobre o conjunto de dados

As variáveis do conjunto de dados são:

checking_status: o status da conta corrente do requerente, que pode ser "0 <= x < 200 DM", ">= 200 DM", "sem conta corrente" ou "outros".

duration: duração do crédito em meses.

credit_history: histórico de crédito do requerente, que pode ser "crédito existente quitado pontualmente", "crédito existente quitado pontualmente neste banco", "todos os créditos quitados pontualmente", "crédito existente não quitado pontualmente", "nenhum crédito existente".

purpose: propósito do crédito, como "carro novo", "carro usado", "móveis/equipamentos", "rádio/TV", "eletrodomésticos", "reparos", "educação", "férias", "treinamento/projeto".

credit_amount: montante do crédito solicitado em DM (marcos alemães).

savings_status: status da poupança do requerente, que pode ser "<100 DM", "100 <= x < 500 DM", "500 <= x < 1000 DM", ">= 1000 DM", "desconhecido/nenhum".

employment: status de emprego/atual emprego do requerente, que pode ser "desempregado", "<1 ano", "1 <= x < 4 anos", "4 <= x < 7 anos" ou ">= 7 anos".

installment_commitment: quantidade de renda disponível após o pagamento de outras prestações, em porcentagem.

personal_status: status pessoal e sexo do requerente, que pode ser "masculino solteiro", "masculino casado/divorciado", "feminino solteira" ou "outros".

other_parties: outros devedores/garantidores presentes no contrato de crédito, que podem ser "nenhum", "co-requerente" ou "garantidores".

residence_since: tempo de residência atual do requerente em anos.

property_magnitude: tamanho da propriedade do requerente, que pode ser "nenhum", "carro", "seguro de vida/planos de poupança", "imóvel".

age: idade do requerente em anos.

other_payment_plans: outros planos de pagamento existentes, que podem ser "nenhum", "bens de consumo", "banco".

housing: status de moradia do requerente, que pode ser "próprio", "alugado", "de graça".

existing_credits: quantidade de créditos já existentes em bancos.

job: status profissional do requerente, que pode ser "desempregado/não-existente", "<1 ano", "1 <= x < 4 anos", "4 <= x < 7 anos", ">= 7 anos".

num_dependents: número de pessoas dependentes financeiramente do requerente.

own_telephone: indica se o requerente possui telefone próprio ou não.

foreign_worker: indica se o requerente é ou não um trabalhador estrangeiro.

### Etapas para resolução do problema:

**1 - Preparação dos dados:**  Antes de treinar um modelo, os dados devem ser preparados. Isso inclui a remoção de dados ausentes, a codificação de variáveis categóricas e a divisão dos dados em conjuntos de treinamento e teste.

**2 - Treinamento do modelo:** O próximo passo é treinar o modelo Naive Bayes com o conjunto de treinamento. O modelo é treinado usando a probabilidade condicional das características de cada classe. Neste caso, a classe pode ser 'good' ou 'bad'

**3 - Teste do modelo:** Depois que o modelo é treinado, ele é testado usando o conjunto de teste. O modelo faz previsões para cada exemplo no conjunto de teste e as compara com os rótulos verdadeiros. O desempenho do modelo é avaliado usando métricas como a acurácia e a matriz de confusão.

**4 - Avaliação do modelo:** Depois de testar o modelo, o desempenho é avaliado para garantir que ele esteja funcionando bem. Isso pode ser feito ajustando os parâmetros do modelo, como o limiar de classificação, ou usando técnicas de validação cruzada para avaliar o desempenho do modelo em diferentes conjuntos de dados.

**5- Previsão de novos dados:** Após o modelo ser considerado adequado, ele pode ser usado para avaliar a probabilidade de um novo cliente ser um bom ou mau pagador com base em suas características.
