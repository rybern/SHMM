{-# LANGUAGE ForeignFunctionInterface, OverloadedLists #-}
module Main where

import Data.Vector (Vector)
import qualified Data.Vector as Vector
import SHMM

n_states = 2
emissions = [
    [0.9, 0.2]
  , [0.9, 0.2]
  , [0.1, 0.8]
  , [0.9, 0.2]
  , [0.9, 0.2]
  ]
triples = [
    (0, 1, 0.5)
  , (0, 2, 0.5)
  , (1, 1, 0.7)
  , (1, 2, 0.3)
  , (1, 3, 0.5)
  , (2, 1, 0.3)
  , (2, 2, 0.7)
  , (2, 3, 0.5)
  ]
permutation = [0, 1, 2]

main :: IO ()
main = do
  summed <- shmmSummed n_states triples emissions permutation
  full <- shmmFull n_states triples emissions permutation
  let fullSummed' n = Vector.sum $ Vector.map (Vector.! n) full
      fullSummed = Vector.map fullSummed' [0..(Vector.length (Vector.head full) - 1)]
  putStrLn "Final post:"
  print full
  print summed
  print fullSummed
  --forM_ posterior $ \row ->
    --print row
