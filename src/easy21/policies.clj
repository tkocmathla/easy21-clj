(ns easy21.policies
  (:require [easy21.environment :as env]))

(defn e-greedy [S Q N]
  (let [n (reduce + (vals (filter (comp #{S} first key) N)))
        e (/ 100 (+ 100 n))
        qs (filter (comp #{S} first key) Q)]
    (if (or (< (rand) e) (empty? qs))
      (rand-nth env/all-actions)
      (second (key (apply max-key val qs))))))

