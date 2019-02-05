(ns easy21.policies
  (:require [easy21.environment :as env]))

(defn e-greedy [S Q N]
  (let [n (+ (N [S :hit] 0) (N [S :stick] 0))
        e (/ 100 (+ 100 n))
        qs (merge (select-keys Q [S :hit]) (select-keys Q [S :stick]))]
    (if (or (< (rand) e) (empty? qs))
      (rand-nth env/all-actions)
      (second (key (apply max-key val qs))))))
