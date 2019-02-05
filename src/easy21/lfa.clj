(ns easy21.lfa
  "Linear function approximation"
  (:require
    [easy21.environment :as env]
    [clojure.core.matrix.operators :as ops])
  (:use
    [clojure.core.matrix]
    [clojure.core.matrix.random]))

(set-current-implementation :vectorz)

(def features
  (reduce
    (fn [m [[d p :as S] A]]
      (let [fv (for [x [(<= d 4) (<= 4 d 7) (<= 7 d 10)]
                     y [(<= p 6) (<= 4 p 9) (<= 7 p 12) (<= 10 p 15) (<= 13 p 18) (<= 16 p 21)]
                     z [(= A :hit) (= A :stick)]]
                 (if (and x y z) 1 0))]
        (assoc m [S A] (array fv))))
    {} env/all-states))

(defn Q [S A w]
  (dot (features [S A]) w))

(defn e-greedy [{:keys [S w epsilon]}]
  (if (< (rand) epsilon)
    (rand-nth env/all-actions)
    (max-key #(Q S % w) :hit :stick)))

(defn sarsa
  [{:keys [S A E alpha gamma lambda w] :as m}]
  (let [[S* R] (env/step S A)
        A* (when-not (env/end? S*) (e-greedy m))
        delta (if-not (env/end? S*)
                (+ R (* gamma (Q S* A* w)) (- (Q S A w)))
                (- R (Q S A w)))
        X (features [S A])
        E (ops/+ (ops/* gamma lambda E) X)]
    (-> m
        (assoc :E E)
        (assoc :w (ops/+ (ops/* alpha delta E) w))
        (assoc :S S*)
        (assoc :A A*))))

(defn episode [init]
  (->> {:S [(Math/abs (env/draw)) (Math/abs (env/draw))]
        :A :hit
        :E (fill (new-vector 36) 0)
        :alpha 0.01
        :epsilon 0.05
        :gamma 1}
       (merge init)
       (iterate sarsa)
       (drop-while (comp not env/end? :S))
       first))

;; -----------------------------------------------------------------------------

(comment
  (require
    '[clojure.data.csv :as csv]
    '[clojure.edn :as edn]
    '[clojure.java.io :as io]
    '[clojure.pprint :refer [pprint]]
    '[clojure.string :as string])

  (defn dump-q [episodes lambda w]
    (->> (map (fn [[s a]] [[s a] (Q s a w)]) env/all-states)
         (map (fn [[[s a] q]] [s q]))
         (group-by first)
         (map (fn [[[d p] xs]] [d p (->> xs (map last) (apply max))]))
         (map #(string/join #"," %))
         (string/join "\n")
         doall
         (spit (format "Q-lfa-%se-%sl.csv" episodes (double lambda)))))

  (defn mse [Q* w]
    (/ (reduce + (map (fn [[s a]] (Math/pow (- (Q s a w) (Q* [s a])) 2)) env/all-states))
       (count env/all-states)))

  (let [episodes (int 1e3)
        Q* (edn/read-string (slurp "Q.edn"))]
    (with-open [writer (io/writer "Q-lfa-mse.csv")]
      (->> (for [lambda (range 0 11/10 1/10)
                 :let [init-w (sample-uniform 36)
                       {:keys [w]} (->> {:lambda lambda, :w init-w} (iterate episode) (take episodes) last)]]
             (do (dump-q episodes lambda w)
                 [lambda (mse Q* w)]))
           (csv/write-csv writer)))))
