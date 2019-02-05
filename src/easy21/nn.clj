(ns easy21.nn
  "Neural network function approximation"
  (:require
    [clojure.core.matrix.operators :as ops]
    [clojure.data.csv :as csv]
    [clojure.edn :as edn]
    [clojure.java.io :as io]
    [cortex.experiment.classification :as classification]
    [cortex.experiment.train :as train]
    [cortex.nn.execute :as execute]
    [cortex.nn.layers :as layers]
    [cortex.nn.network :as network]
    [cortex.util :as util]
    [easy21.environment :as env]
    [taoensso.tufte :refer [defnp profile] :as tufte])
  (:use
    [clojure.core.matrix]
    [clojure.core.matrix.random]))

(set! *warn-on-reflection* true)

(defn argmax [xs]
  (->> xs
       (map-indexed vector)
       (apply max-key second)
       first))

(def network-description
  [(layers/input 18 1 1 :id :x)
   (layers/linear->relu 18)
   (layers/linear 2 :id :y, :loss :mse-loss)])

(def context
  (execute/compute-context :datatype :float, :backend :cpu))

(def features
  (reduce
    (fn [m [[d p :as S]]]
      (let [fv (for [x [(<= d 4) (<= 4 d 7) (<= 7 d 10)]
                     y [(<= p 6) (<= 4 p 9) (<= 7 p 12) (<= 10 p 15) (<= 13 p 18) (<= 16 p 21)]]
                 (if (and x y) 1 0))]
        (assoc m S (array fv))))
    {} env/all-states))

(defn Q [nn S]
  (let [X [{:x (features S)}]
        [{:keys [y]}] (execute/run nn X :context context, :batch-size 1)]
    y))

(defn e-greedy [S nn epsilon]
  (if (< (rand) epsilon)
    (rand-nth env/all-actions)
    (env/all-actions (argmax (Q nn S)))))

;; -----------------------------------------------------------------------------
;; Training

(defn train [nn X y]
  (let [data [{:x X, :y y}]
        {:keys [network]} (execute/train nn data :context context, :batch-size 1)]
    network))

(defn sarsa
  [{:keys [S A nn epsilon gamma] :as m}]
  (let [[S* R] (env/step S A)
        A* (when-not (env/end? S*) (e-greedy S* nn epsilon))
        target (if-not (env/end? S*)
                 (+ R (* gamma (apply max (Q nn S*))))
                 (- R (apply max (Q nn S))))]
    (-> m
        (assoc :nn (train nn (features S) (assoc [0 0] (argmax (Q nn S)) target)))
        (assoc :S S*)
        (assoc :A A*)
        (assoc :R R))))

(defn episode [init]
  (->> {:S [(Math/abs ^int (env/draw)) (Math/abs ^int (env/draw))]
        :A :hit
        :gamma 0.99
        :i (inc (:i init))
        :epsilon (/ 1 (+ (/ (:i init) 100) 100))}
       (merge init)
       (iterate sarsa)
       (drop-while (comp not env/end? :S))
       first))

;; -----------------------------------------------------------------------------
;; Inference

(defn infer
  [{:keys [S nn] :as m}]
  (let [A (env/all-actions (argmax (Q nn S)))
        [S* R] (env/step S A)]
    (assoc m :S S* :R R)))

(defn play [nn]
  (->> {:S [(Math/abs ^int (env/draw)) (Math/abs ^int (env/draw))]
        :nn nn}
       (iterate infer)
       (drop-while (comp not env/end? :S))
       first))

;; -----------------------------------------------------------------------------

(require
  '[clojure.data.csv :as csv]
  '[clojure.edn :as edn]
  '[clojure.java.io :as io]
  '[clojure.pprint :refer [pprint]]
  '[clojure.string :as string])

(set! *print-length* nil)

; play a bunch of games and summarize the outcomes
#_
(let [nn (util/read-nippy-file "easy21.nippy")]
  (frequencies
    (for [_ (range 1e3)]
      (:R (play nn)))))

; probe the network for the distribution of predictions across all states
#_
(let [nn (util/read-nippy-file "easy21.nippy")]
  (frequencies
    (for [d (range 1 11), p (range 1 22)]
      (env/all-actions (argmax (Q nn [d p]))))))

; compare NN to optimal policy from MC
#_
(let [Q* (->> (edn/read-string (slurp "Q.edn"))
              (map (fn [[[s a] v]] [s a v]))
              (group-by first)
              (map (fn [[s xs]] [s (second (apply max-key last xs))]))
              (into {}))
      nn (util/read-nippy-file "easy21.nippy")]
  (frequencies
    (for [d (range 1 11), p (range 1 22)]
      (= (Q* [d p]) (env/all-actions (argmax (Q nn [d p])))))))

; train network
#_
(time
  (->> {:i 0, :nn (network/linear-network network-description)}
       (iterate episode)
       (take 1e4)
       last
       :nn
       (util/write-nippy-file "easy21.nippy")))
