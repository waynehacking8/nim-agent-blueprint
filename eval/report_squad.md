# Agentic-RAG eval — retrieval, hallucination gating, judge calibration, latency

Corpus: 100 SQuAD 2.0 dev passages · 100 answerable + 100 unanswerable questions (SQuAD 2.0's unanswerable questions are crowdworker-written adversarial near-misses: the on-topic paragraph IS in the corpus but does not contain the answer). Self-hosted on H100: vLLM generator + Ollama embeddings. k=3, temp=0. All percentages carry 95% Wilson confidence intervals.

## Headline

| metric | value |
|---|---|
| retrieval recall@3 (answerable) | 86% (86/100, 95% CI 78%–91%) |
| answer accuracy — substring (guarded, answerable) | 71% (71/100, 95% CI 61%–79%) |
| answer accuracy — LLM judge (guarded, answerable) | 72% (72/100, 95% CI 63%–80%) |
| hallucination on unanswerable — **guarded** generator | **49% (49/100, 95% CI 39%–59%)** |
| hallucination on unanswerable — **unguarded** generator | **79% (79/100, 95% CI 70%–86%)** |

The guarded prompt ("answer only from context, else say you don't know") is the first line of defense. The ablation shows what happens without it.

![Hallucination ablation](report_squad_ablation.png)

## The validate() groundedness gate as a safety net (unguarded run)

Second line of defense: an LLM-as-judge checks each answer against the retrieved context and blocks unsupported ones. Scored on the unguarded run, where hallucinations exist to catch.

### Self judge (same model/endpoint as the generator)

| | gate blocks | gate passes |
|---|---|---|
| hallucinated (should block) | TP=30 | FN=65 |
| grounded/abstained (should pass) | FP=8 | TN=97 |

- precision **79% (30/38, 95% CI 64%–89%)** · recall **32% (30/95, 95% CI 23%–41%)** · F1 **0.45**
- residual hallucination on unanswerable after gating: **52% (52/100, 95% CI 42%–62%)** (down from 79% (79/100, 95% CI 70%–86%))

### Cross-family judge (llama-3.1-8b)

| | gate blocks | gate passes |
|---|---|---|
| hallucinated (should block) | TP=45 | FN=50 |
| grounded/abstained (should pass) | FP=18 | TN=87 |

- precision **71% (45/63, 95% CI 59%–81%)** · recall **47% (45/95, 95% CI 38%–57%)** · F1 **0.57**
- residual hallucination on unanswerable after gating: **38% (38/100, 95% CI 29%–48%)** (down from 79% (79/100, 95% CI 70%–86%))

### Self vs cross-family judge — the shared-blind-spots test

Hypothesis: a judge that is the same model that hallucinated misses the same facts it got wrong (shared blind spots), so an independent judge from a different model family should catch more.

| judge | precision | recall | F1 |
|---|---|---|---|
| self (qwen3-8b) | 79% | 32% | 0.45 |
| cross-family (llama-3.1-8b) | 71% | 47% | 0.57 |

Recall delta: **+16 points**, and the comparison is *paired* (both judges score the same answers — temp=0 makes the generations identical), so an exact McNemar test applies: of the 23 hallucinations caught by exactly one judge, the cross-family judge caught **19** vs the self judge's **4** — **p = 0.0026**. Cross-model detection literature (e.g. FINCH-ZK, arXiv:2508.14314: detection F1 improved by 6–39% on FELM from cross-model consistency) predicts this gain when blind spots, not judging ability, are the bottleneck.

The equally important honest number: **46** of the 95 hallucinations were caught by *neither* 8B judge — two small same-size judges still share most blind spots with each other. The next rungs are a larger judge model, a judge panel (PoLL), or retrieval-grounded verification.

![Gate comparison](report_squad_gates.png)

## Per-hop latency budget (mean seconds, guarded)

| plan | retrieve | generate | validate | total |
|---|---|---|---|---|
| 0.22 | 0.181 | 0.34 | 0.16 | 0.90 |

## Per-question detail

| question | answerable | recall@k | correct (LLM) | unguarded halluc | self gate | x-gate |
|---|---|---|---|---|---|---|
| What aspects of life does Islamism seek to int | True | True | True | False | False | False |
| When did the Black Death spread into India? | False | None | False | False | False | True |
| When was the Latin version of the word Norman  | True | False | False | False | True | True |
| In what year did William the Silent issue his  | False | None | False | True | True | True |
|  What was Tugh Temur not known for? | False | None | False | False | False | True |
| What equates to a squared integer according to | True | True | True | False | False | False |
| In the shortest building in downtown Jacksonvi | False | None | False | True | False | False |
| What effect would happen if P is ultimately pr | False | None | False | True | False | False |
| To what can the use of prolonged breathing of  | True | True | True | False | False | False |
| Who wrote "On the Computational Complexity of  | False | None | False | False | False | False |
| What is the most frequently employed type of r | True | True | True | False | False | False |
|  How did the Yuan come to have the 8 schools o | False | None | False | True | False | False |
| Vice President Agnew describes Civil disobedie | True | True | True | False | False | False |
| What is Warsaw's symbol? | True | True | True | False | False | False |
| In 1940, what percentage of the population in  | True | True | True | False | False | False |
| Who included 1 as the first prime number in th | True | True | False | False | False | False |
| The rotational inertia of planet Mars is what  | False | None | False | True | True | True |
| In what year was Fort Coligny built? | False | None | False | True | True | False |
| Which country's cars became more highly sought | True | True | True | False | False | False |
| What number did Henri Lebesgue not regard as a | False | None | False | True | True | False |
| What action did the US plan to take in 2004 du | False | None | False | True | False | False |
| What organization is responsible for their own | False | None | False | True | False | False |
| How did the Los Angeles Times define southern  | False | None | False | True | False | True |
| What beds are nitrogen-saturated?  | False | None | False | True | False | False |
| What can exposure to partial gas pressures gre | False | None | False | True | False | True |
| When did General Sejm make Vilnius its permane | False | None | False | False | False | False |
| What is another type of public key cryptograph | True | True | True | False | False | False |
| What is the process of removing trees from a f | True | True | True | False | False | False |
| When did Setanta Sports say it would launch as | False | None | False | False | False | False |
| What were the first two destinations of Huguen | True | True | True | False | False | False |
| How did the Yuan come to have the 4 schools of | True | True | True | True | False | False |
| What can the IPCC's report deadlines cause to  | True | True | True | False | False | False |
| What does the IPCC rely on for research? | True | False | False | False | False | False |
| What term do Islamists think should be applied | True | True | False | False | False | False |
| What makes day length constant on Earth? | True | True | True | False | False | False |
| What paper is commonly considered the bellweth | True | True | True | False | False | False |
| What percentage of Pinedale is black? | False | None | False | True | False | False |
| When was the Dutch Revolt? | True | True | True | True | False | False |
| What channels were removed from the network in | True | True | False | False | False | False |
| When did the General Sejm make Warsaw it's per | True | True | True | False | True | False |
| Besides the study of prime numbers, what gener | False | None | False | False | False | True |
| What individuals were responsible for authorin | True | False | False | True | False | False |
| What are the Sky Q mini set top boxes never ab | False | None | False | True | False | False |
| What is a popular strolling destination for ph | False | None | False | True | False | True |
| When did the high school education movement oc | True | True | True | True | False | False |
| What is a particular problem in biology that w | True | True | True | False | False | False |
| What name comes from the English words Normans | False | None | False | True | False | True |
| Besides the study of prime numbers, what gener | True | False | False | False | True | True |
| When did income inequality begin to decrease i | False | None | False | True | False | False |
| How many French colonists were lost by British | False | None | False | False | False | False |
| What did Standard & Poor recommend to slow eco | False | None | False | True | False | False |
| What revealed the intentions of the British in | False | None | False | True | False | True |
| How much potential economic growth could the U | False | None | False | True | False | True |
| Which country was thinking about going to war  | True | True | True | True | False | False |
| When was the University of Chicago established | True | True | True | False | False | False |
| Who benefits from the research carried out by  | False | None | False | True | True | True |
| How many homes had BSkyB's direct-to-home sate | True | True | True | False | False | False |
| When did BSkyB announce it's intention to repl | True | True | True | False | False | False |
| Which country was worried that the US would in | True | False | False | True | False | True |
| How large was the audience BSkyB said they cou | True | True | True | True | False | False |
| What author argues pitching the conscience ver | True | True | True | False | False | False |
| When did Augustus die? | True | True | True | False | False | False |
| What is the meaning of polynomial-space reduct | False | None | False | True | True | False |
| What impact did the high school education move | False | None | False | True | False | False |
| What makes the method of primality more effici | False | None | False | True | False | True |
| What is involved in a review of prescribed med | True | True | True | True | False | False |
| Who joined Norman forces in the destruction of | False | None | False | True | False | False |
| What is polish for "mermaid"? | True | True | True | False | False | False |
| What number did early Greeks not regard as a t | True | True | True | False | False | True |
| What is known as swing pressure adsorption? | False | None | False | True | False | False |
| What is the original meaning of the word Norma | True | False | False | True | True | True |
| What did the Los Angeles Times add to the defi | False | None | False | True | False | False |
| What seminal paper is commonly considered the  | False | None | False | False | True | True |
| What is the name of Sky Q's broadband router? | True | True | True | False | False | False |
| What was the name of the infamous German Heroi | False | None | False | True | False | False |
| What did the people of Rome accept as the only | False | None | False | True | True | True |
| If P is ultimately proven to be equal tot NP,  | True | True | True | True | False | False |
| What was the taxman's political philosophy? | False | None | False | False | False | False |
| The concept of inertia can explain the tendenc | False | None | False | True | False | True |
| What districts does downtown Santa Ana include | False | None | False | False | False | False |
| What does a writer for the International Crisi | True | True | True | False | False | False |
| How many schools of medicine were recognized i | False | None | False | False | False | False |
| What did Standard & Poor recommend to speed ec | True | True | True | False | False | False |
| What are the Sky Q mini set top boxes able to  | True | True | True | False | False | False |
| When did income inequality begin to increase i | True | True | True | False | False | False |
| When was the Sierra Sky Park Airport formed? | True | True | True | False | False | False |
| What year did BSkyB and Virgin Media have an a | False | None | False | True | True | True |
| Assessing the Amazon rain forest was restricte | False | None | False | True | True | True |
| What is the name of Sky Q's dial-up router? | False | None | False | True | False | False |
| What academy did Tugh Temur found? | True | True | True | False | False | False |
|  Who did Duke Yansheng Kong Duanyou stay with? | False | None | False | True | True | True |
| What impact did the high school education move | True | True | True | False | False | False |
| What is one example of what a clinical pharmac | True | True | False | True | False | False |
| What can happen when breathing in oxygen with  | False | None | False | True | True | True |
| The Rhine and what other river were accepted a | True | True | True | False | False | False |
| What commemorates Old Town's heroic history? | False | None | False | True | False | False |
| What is tuition for 2012 - 13 year at Harvard? | True | True | False | True | False | False |
| How many schools of medicine were recognized i | True | True | False | False | False | False |
| Where and when did the investigation of the pl | True | True | True | False | False | False |
| How many French colonists were gained by Briti | True | True | True | False | False | True |
| Which county is developing its business center | True | True | True | False | False | False |
| What archdiocese is Market Square the seat of? | False | None | False | True | False | False |
| When did the Jin dynasty begin? | True | True | True | False | False | False |
| When did Augustus find Rome? | False | None | False | True | True | True |
| Who did Warsaw serve as the seat for in 1529? | True | True | True | False | False | False |
| What is not an example of what a clinical phar | False | None | False | True | True | False |
| What was Thoreau's punishment for not paying h | True | False | False | False | False | False |
| What is Polish for "female"? | False | None | False | True | True | True |
| What is a non-Islamic revival movement? | False | None | False | True | True | True |
| What was the white population in 1942 in Fresn | False | None | False | True | False | True |
| During what time period did income inequality  | False | None | False | True | False | False |
| What are two cars with V8 engines that were mo | False | None | False | True | False | False |
| When did Setanta Sports say it would launch as | True | True | False | False | False | False |
| Which sized cars were the least demanded cars  | True | True | False | False | False | False |
| What is the least used type of reduction? | False | None | False | False | False | False |
| How many homes had BSkyB's direct-to-home sate | False | None | False | True | True | True |
| What did families with incomes below $38,000 p | False | None | False | True | True | True |
| Who created the nation's first aviation commun | True | True | True | False | False | False |
| What percentage of oxygen will a zeolite sieve | True | True | True | False | False | False |
|  When did the Germanic tribes not claim territ | False | None | False | True | True | True |
| What was the name of the infamous German Gesta | True | True | True | False | False | False |
| Over how many species of trees can be found in | True | True | True | False | False | False |
| What is the most elemental way to test the pri | False | None | False | True | False | False |
| The city of Pritzker is home to which Universi | False | None | False | True | False | True |
| What was considered responsible for the black  | True | True | False | False | False | False |
| Who created all of the nation's aviation commu | False | None | False | True | False | False |
| What county does the rapidly developing downto | False | None | False | True | False | False |
| What health condition can deep sea diving caus | True | True | True | False | False | False |
| What is another notable university in Warsaw a | True | True | False | False | False | False |
| Who did the Turks take up service with? | False | None | False | False | False | True |
| What was the name of the Norman castle? | True | True | True | False | False | False |
| Where were the Germanic tribes not originally  | False | None | False | True | False | False |
| Civil disobedience has been argued in more rec | True | True | True | False | False | False |
| During what time period did income inequality  | True | True | True | False | False | False |
| Who included 1 as the first prime number in th | False | None | False | True | False | True |
|  When did the Jip dynasty begin? | False | None | False | True | False | False |
| What nationality was Pierre L'Oyseleur? | False | None | False | True | False | True |
| What archdiocese is Warsaw the seat of? | True | True | True | False | False | False |
| What makes the method of trial division more e | True | False | False | True | True | True |
| What garden was formally only for running? | False | None | False | True | True | True |
| What term don't Islamists think should be appl | False | None | False | False | True | True |
| What mechanism can be used to make oxygen? | True | True | False | False | False | False |
| In which year did the newspaper define souther | True | True | True | False | False | False |
| What ethnic neighborhood in Fresno had primari | True | True | False | False | False | False |
| The process of growing more trees in the fores | False | None | False | True | True | True |
| How much potential economic growth could the U | True | True | False | False | False | False |
| In what year did Alexandre Yersin discover the | False | None | False | True | False | True |
| Who did Duke Yansheng Kong Duanyou flee with? | True | True | True | False | False | False |
| What channels were always available on the net | False | None | False | True | False | True |
| In what year did the Black Eagle Brewery chang | False | None | False | True | True | True |
| Who was the leader when the Franks entered the | True | False | False | False | False | False |
| What British mathematician took pride in doing | True | True | True | False | False | False |
| What is one type of public key cryptography al | True | True | True | False | False | False |
| What is an Islamic revival movement? | True | True | True | False | False | False |
| What did the Vilnius formally establish in 157 | False | None | False | True | True | True |
| What concept explains why objects continue in  | True | True | True | False | False | False |
| What standards did American cars create in the | False | None | False | False | False | True |
| What distinction does the Bank of America Towe | True | False | False | False | False | True |
| What is another type of private key cryptograp | False | None | False | True | False | True |
| What was Tugh Temur known for? | True | True | True | False | False | False |
|  What does a writer for the International Cris | False | None | False | True | True | True |
| Which newspaper defined southern California? | True | False | False | False | False | True |
| What else were families with incomes below $38 | False | None | False | True | False | False |
| What is never involved in a review of prescrib | False | None | False | True | False | False |
| What garden was formally only for royalty? | True | True | True | False | True | True |
| What is Sigilium's symbol? | False | None | False | True | False | True |
| How many Huguenots were part of the group that | False | None | False | True | False | False |
| What commemorates Warsaw's heroic history? | True | True | True | True | False | False |
| What is one type of private key cryptography a | False | None | False | False | True | True |
|  What academy did Tugh Temur destroy? | False | None | False | True | False | False |
| Where are international corporations headquart | True | True | False | False | False | False |
| When was the deportation of Acadians? | True | True | True | False | False | False |
| What kind of university is the University of C | True | True | True | False | False | False |
| Building a 617 m tall? | False | None | False | True | True | True |
| What year did BSkyB and Virgin Media have a di | True | True | True | False | False | False |
| When was the French version of the word Norman | False | None | False | False | True | True |
| What had Vice President Agnew always suffered  | False | None | False | True | False | True |
| The Bank of America Tower was previously known | True | False | False | True | True | True |
| Nearly 7,000 students are enrolled where? | False | None | False | True | True | True |
| Who did the Dutch fight in the Dutch Revolt? | True | True | True | True | False | False |
| When did the middle school education movement  | False | None | False | False | False | True |
| What early Huguenot Church was established in  | True | False | False | False | False | False |
| What was Warsaw's Market Square listed as in 1 | False | None | False | True | False | False |
| What British mathematician took pride in doing | False | None | False | True | False | False |
| When was the French colony in modern day Brazi | True | True | True | False | False | False |
| When did BSkyB announce it's intention to impr | False | None | False | True | False | False |
| How many French colonists weren't gained by Br | False | None | False | False | False | False |
| In what year did Huguenot refugees first start | False | None | False | True | True | True |
| When did the Sierra Sky Park fall out of use? | False | None | False | False | False | False |
| What is the total cost of attendance in 2012-1 | True | True | True | False | False | False |
| What was the government the final judge of? | False | None | False | True | False | False |
| What had happened to Vice President Agnew's st | False | None | False | True | False | False |
|  What aspects of life does Islamism not seek t | False | None | False | False | False | False |
| When did the Germanic tribes claim territory i | True | True | True | False | False | False |
| Acessing the Amazon rainforest was restricted  | True | True | True | False | False | False |
| When was the charter for this church signed? | True | False | False | False | False | True |
| Where were the Germanic tribes originally loca | True | True | True | False | False | False |
| What is the most elemental way to test the pri | True | True | True | False | False | False |
| What is one of the least important open questi | False | None | False | True | False | True |
| How small was the audience BSkyB said they cou | False | None | False | True | True | True |

