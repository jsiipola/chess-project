Moi,

tässä vähän sekalaista selvennystä.  Aloitin projektin noin vuosi sitten. Projekti on ollut mielessä, 
mutta toiminta on ollut epäaktiivista pitkään. Nyt aktivoiduin, päätin samaan syssyyn kutsua muita mukaan projektiin.
Kutsuin mukaan projektin avustajiksi (collaborator) Tuomaksen, Siirin ja Ossin. Tervetuloa!

Tähän mennessä projekti on pohjautunut pitkälle siihen mitä Matthew Lai kirjoitti gradussaan:
https://arxiv.org/abs/1509.01549, Giraffe: Using Deep Learning to Play Chess.
Jos siis haluat selkeän referenssin siihen mitä olen ollut tekemässä niin voit lukea tuon
artikkelin eli gradun. Matthew Lai on gradunsa jälkeen tehnyt töitä Googlella niissä AlphaGo-projekteissa.

Mitä on tehty: 

(1) Alkeellinen puuhaku (minimax, minimax alpha-beta karsimisella, https://www.youtube.com/watch?v=l-hh51ncgDI)
joka löytyy projektin kirjastosta (projectLib) skriptistä depth_search.py. Sinne saa kirjoittaa parempia 
versioita puuhausta.

(2) Alkeellinen ominaisuus vektori (tai atribuutti vektori/tensori, eng. feature representation). Tämä
vektori on se joka syötetään neuroverkolle ns. input layeriin, eli vektori jonka neuroverkko kuvaa output
layerille (en näköjään jaksa kääntää kaikkea, huoh.) Tämä löytyy projekti kirjastosta skriptistä jonka nimi
on feature_representation.py. Ajatus näissä on tietenkin luoda tietynlaisia luokkia jonka alle kasaa sitten
toiminnallisuutta. Esim. puuhaulle on oma luokkansa, samoin feature representationille (ominaisuus vektorille).

(3) En ole tehnyt, mutta löysin ja ihan hyväksi havaitsin pythonille kirjoitetun shakki kirjaston. Siinä
kirjastossa on paljon hyödyllistä toiminnallisuutta joka helpottaa tälläisen algoritmin kirjoittamista.
Ei se ole ihan täydellinen, mutta luo kai ihan ok pohjan jonka päälle voi kasata lisää. 
https://python-chess.readthedocs.io/en/latest/

(4) main.py on tällä hetkellä lähinnä sellainen josta käsin olen testannut miten asiat toimivat.
Mitään lopullisen valmista algoritmia ei ole vielä olemassa.

Mitä pitäisi tehdä:

(1) Isoin asia johon ei ole tehty vielä mitään on neuroverkon koulutus. Sitä varten ei ole vielä tehty
yhtään mitään. Esim. minkälainen loss funktio? Tarvitaanko koulutusdataa?

(2) Puuhakua täytyy parantaa. Tällä hetkellä se ei oikein sovellu shakkiin, tai sopiihan se, mutta
puuhaku ei osaa sanoa että oliko haun katkaiseminen juuri tähän kohtaan järkevää: Jos kuningatar on juuri
syönyt sotilaan on päästy pisteissä edelle, mutta jos vastustaja saa seuraavalla vuorolla syödä kuningattaren
niin silloin tilanne onkin todella huono, kuningatar-sotilas on huono vaihtokauppa. Huono puuhaku ei osaa 
ottaa tälläistä huomioon. Toinen näkökulma on että puuhaussa olisi parempi pystyä ennakoimaan sitä mitkä 
haarat kannattaisi seuloa läpi ja mitkä ei.

(3) Siinä miten peli saadaan vietyä läpi kun algoritmi pelaa esimerkiksi itseään vastaan, siinä on varmasti
vielä jotain tekemistä.

(4) Käyttöliittymä jolla ihminen voisi pelata algoritmia vastaan.

(5) Ominaisuus vektorin kehittäminen. Tuskin akuutein asia, mutta voi olla tärkeä näkökulma kunnolla toimivan
algoritmin luomisessa.


Lopuksi vielä lyhyt katsaus kirjaston skripteihin.

__init__.py
Tekee kansion skripteistä Python paketin.

depth_search.py
Määrittelee luokan joka hallinnoi puuhakujen toiminnallisuutta.

feature_representation.py
Määrittelee luokan joka ensisijaisesti muodostaa pelitilanteesta ominaisuus vektorin
jonka voi syöttää neuroverkolle, neuroverkolle jonka tehtävä on antaa numeerinen arvio
pelin tilanteesta. Sisältää toisarvoisesti koodia joka on ollut tarpeen kirjoittaa
ensisijaista tehtävää silmällä pitäen. Tämä toisarvoinen koodi on sellaista jolla
voi käsitellä pelilautaa, vähän kai täydennystä siihen pythonin chess kirjastoon.

first-skript.py
Jotain turhaa sceidaa, voinee ehkä poistaa. Mutta ehkä siitä on jollekulle vielä
iloakin. Ensimmäisiä viritelmiä.

get-king-position.py
Samoin kuin edellinen, turhaa skeidaa, mutta kuitenkin hyvä muistutus siitä että
chess kirjasto tarjoaa satunnaisesti erittäin hyödyllisiä toiminnallisuuksia,
toisinaan taas asioita pitää itse koodata. Kuninkaan paikan saa kätevästi valmiilla
metodilla (ei tarvitse itse kirjoittaa metodia).

position_evaluation.py
Ehkä turha luokka, ehkä ei. Jos pelitilanteen arvionti kaipaa toiminnallisuutta
niin sen voi lisätä tähän luokkaan. Nyt siellä on lähinnä se neuroverkko jonka 
pitää suorittaa arvio, ja metodi joka kutsuu neuroverkkoa.

neural-net.py, neural-net2.py, shallow.py
Skriptejä joilla voi luoda neuroverkon. Käsittävät luokan jossa on kaikki tarpeellinen
neuroverkolle. Ei ole mitään järkevää syytä miksi näitä on kolme erilaista.
Kyse on vaan siitä että tää on vielä niin vaiheessa, sori siitä. Voimme
myöhemmin olla järkeviä ja määritellä vain yhden skriptin jossa on tietyn tyyppiset
neuroverkot. Hyväksyn eri skriptit, ja ainakin eri luokat, erilaisille neuroverkko
arkkitehtuureille.

second_script.py
Jotain skeidaa.
