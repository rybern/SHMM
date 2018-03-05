{-# LANGUAGE ForeignFunctionInterface, OverloadedLists #-}
module Main where

import Control.Monad
import Foreign
import Foreign.Storable
import Foreign.C.Types
import Data.Vector (Vector)
import qualified Data.Vector as Vector
import qualified Data.Vector.Storable as Storable
import qualified Data.Vector.Storable.Mutable as Mutable

{-
to do:
  - figure out permutations in libshmm.
    * start by replacing the cwiseProduct(emissions[i]) in forward and backward with more manual iterations over the sparse vectors, without permuting. can use same test for correctness
    * then you can inline the permutation
  - figure out emissions manipulation
    * since emissions will only be read in once, you could have one function early on do all the conversions and leave it as a ((Ptr (Ptr Double) -> IO b) -> IO b, Int, Int)
    * or, just split out the vanilla Emissions -> Emissions and hope it's called once
    * or, add it when reading it in from disk and add it to the spec
    * or, eat the cost of adding it on each call
  - figure out the posterior transposition. issue because posterior is (currently dense) states X events
    * eat the cost ever time
    * use it column major
    * why is C++ using it as a matrix anyway, when emissions isn't? isn't everything else a vector of vectors?
    * ^ this is the answer. all you're doing in forward_backward is setting it row-by-row. keep the interface the same except make it row-major, and use a vector<SparseVector>. should be sparse anyway. why did you do it this way?
    * actually, shouldn't the interface be sparse too?

  - general todo: investigate truncating 0s. that might be a vital knob.
-}

shmm ::
  -- The number of states, not including tokens
  Int ->
  -- Triples representing the transition distribution.
  -- Index 0 is reserved for the start token, other states start from 1.
  -- Column can go up to n_states+1, which represents the end token.
  [(Int, Int, Double)] ->
  -- Vector of observation distributions, one for each time point.
  -- Each distribution is n_obs long, which can be less than n_states.
  Emissions ->
  -- Vector to encode the permutation of states -> obervations.
  -- n_states long with values 0 through n_obs-1.
  Vector Int ->
  -- Returns a vector of posterior distributions over states, one for each event.
  -- n_events vectors of n_states each.
  IO Emissions
shmm n_states' triples emissions permutations = do
  let n_triples = fromIntegral (length triples)
      n_states = fromIntegral n_states'
      n_obs = fromIntegral (Vector.length (Vector.head emissions))
      n_events' = Vector.length emissions
      n_events = fromIntegral (Vector.length emissions)

      --not ideal - where is best to add the 0s?
      emissions' = Vector.map (flip Vector.snoc 0) emissions

  -- initialize memory for the posterior, so we know where to read
  post <- Mutable.unsafeNew (n_events' * (n_states' + 1))

  -- call the C++ function, with temporary Ptrs to avoid a space leak
  withTripleArray triples $ \tsPtr ->
    withEmissions emissions' $ \esPtr ->
      Storable.unsafeWith (Storable.convert permutations) $ \permPtr ->
        Mutable.unsafeWith post $ \postPtr ->
          c_shmm n_triples tsPtr n_states n_obs esPtr n_events permPtr postPtr

  -- post is a contiguous mutable vector, convert it to an immutable 2D vector
  postFrozen <- Storable.freeze post
  let postEmissions = flip Vector.map [0 .. (n_states'-1)] $ \ix ->
        Vector.convert $ Storable.slice (n_events' * ix) n_events' postFrozen

  --not ideal, not sure if it's avoidable - Eigen uses col-major
  let postEmissions' = flip Vector.map [0..(n_events'-1)] $ \ix -> Vector.map (Vector.! ix) postEmissions

  return postEmissions'

-- C++ signature:
-- int shmm(int n_triples, DTriple *triples, int n_states, int n_obs, double **emissions_aptr, int n_events, int *permutation_, double *posterior_arr )
foreign import ccall "_Z4shmmiP7DTripleiiPPdiPiS1_"
  c_shmm :: CInt -> Ptr DTriple -> CInt -> CInt -> Ptr (Ptr Double) -> CInt -> Ptr Int -> Ptr Double -> IO CInt

data DTriple = DTriple CInt CInt CDouble
type Emissions = Vector (Vector Double)

instance Storable DTriple where
  sizeOf _ = 16
  alignment = sizeOf
  peek ptr = do
    row <- peekByteOff ptr 0
    col <- peekByteOff ptr 4
    val <- peekByteOff ptr 8
    return $ DTriple row col val
  poke ptr (DTriple row col val)= do
    pokeByteOff ptr 0 row
    pokeByteOff ptr 4 col
    pokeByteOff ptr 8 val

buildDTriple :: Int -> Int -> Double -> DTriple
buildDTriple r c v = DTriple (fromIntegral r) (fromIntegral c) (realToFrac v)

withTripleArray :: [(Int, Int, Double)] -> (Ptr DTriple -> IO b) -> IO b
withTripleArray triples = withArray (map (\(r,c,v) -> buildDTriple r c v) triples)

withEmissions :: Emissions -> (Ptr (Ptr Double) -> IO b) -> IO b
withEmissions emissions action = do
  ptrVec <- Vector.mapM (flip Storable.unsafeWith return . Storable.convert) emissions
  let storablePtrVec = Storable.convert ptrVec
  Storable.unsafeWith storablePtrVec action

n_states = 2
emissions = [
    [0.9, 0.2]
  , [0.9, 0.2]
  , [0.1, 0.9]
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
  posterior <- shmm n_states triples emissions permutation
  putStrLn "Final post:"
  forM_ posterior $ \row ->
    print row
