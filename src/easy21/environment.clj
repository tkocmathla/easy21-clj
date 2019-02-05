(ns easy21.environment
  (:require [clojure.math.combinatorics :as combo]))

(def all-actions
  [:hit :stick])

(def all-states
  (combo/cartesian-product
    (combo/cartesian-product (range 1 11) (range 1 22))
    all-actions))

(def end? #{::end})

(defn draw []
  (* (inc (rand-int 10)) (rand-nth [-1 1 1])))

(defn play-dealer [sum]
  (cond
    ; bust
    (or (< sum 1) (> sum 21)) -1
    ; hit
    (< sum 17) (recur (+ sum (draw)))
    ; stick
    :else sum))

(defn step [[dealer player] action]
  (condp = action
    :hit (let [player* (+ player (draw))]
           (if (or (< player* 1) (> player* 21))
             [::end -1]
             [[dealer player*] 0]))

    :stick (let [dealer-sum (play-dealer dealer)]
             (cond
               (> player dealer-sum)
               [::end 1]

               (= dealer-sum player)
               [::end 0]

               (< player dealer-sum)
               [::end -1]))))
