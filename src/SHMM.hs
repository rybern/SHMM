{-# LANGUAGE ForeignFunctionInterface, OverloadedLists #-}
module SHMM
  ( shmmSummed
  , shmmFull
  , shmmSummedUnsafe
  , shmmFullUnsafe
  , shmm
  , VecMat (..)
  ) where

import Control.Monad
import Foreign
import Foreign.Storable
import Foreign.C.Types
import Data.Vector (Vector)
import qualified Data.Vector as Vector
import qualified Data.Vector.Storable as Storable
import qualified Data.Vector.Storable.Mutable as Mutable
import System.IO.Unsafe

{-
to do:
  - figure out permutations in libshmm.
    * start by replacing the cwiseProduct(emissions[i]) in forward and backward with more manual iterations over the sparse vectors, without permuting. can use same test for correctness
    * then you can inline the permutation
  - figure out emissions manipulation
    * since emissions will only be read in once, you could have one function early on do all the conversions and leave it as a ((Ptr (Ptr Double) -> IO b) -> IO b, Int, Int)
    * or, just split out the vanilla VecMat -> VecMat and hope it's called once
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

shmmSummedUnsafe :: Int -> [(Int, Int, Double)] -> VecMat -> Vector Int -> Vector Double
shmmSummedUnsafe n_states' triples emissions permutations' =
  unsafePerformIO $ shmmSummed n_states' triples emissions permutations'

shmmSummed ::
  -- The number of states, not including tokens
  Int ->
  -- Triples representing the transition distribution.
  -- Index 0 is reserved for the start token, other states start from 1.
  -- Column can go up to n_states+1, which represents the end token.
  [(Int, Int, Double)] ->
  -- Vector of observation distributions, one for each time point.
  -- Each distribution is n_obs long, which can be less than n_states.
  VecMat ->
  -- Vector to encode the permutation of states -> obervations.
  -- n_states long with values 0 through n_obs-1.
  Vector Int ->
  -- Returns a vector of posterior distributions over states, one for each event.
  -- n_events vectors of n_states each.
  IO (Vector Double)
shmmSummed n_states' triples emissions permutations' = do
  postFrozen <- shmm n_states' triples emissions permutations' True
  return . Vector.init . Vector.convert $ postFrozen

shmmFullUnsafe :: Int -> [(Int, Int, Double)] -> VecMat -> Vector Int -> VecMat
shmmFullUnsafe n_states' triples emissions permutations' =
  unsafePerformIO $ shmmFull n_states' triples emissions permutations'


shmmFull ::
  -- The number of states, not including tokens
  Int ->
  -- Triples representing the transition distribution.
  -- Index 0 is reserved for the start token, other states start from 1.
  -- Column can go up to n_states+1, which represents the end token.
  [(Int, Int, Double)] ->
  -- Vector of observation distributions, one for each time point.
  -- Each distribution is n_obs long, which can be less than n_states.
  VecMat ->
  -- Vector to encode the permutation of states -> obervations.
  -- n_states long with values 0 through n_obs-1.
  Vector Int ->
  -- Returns a vector of posterior distributions over states, one for each event.
  -- n_events vectors of n_states each.
  IO VecMat
shmmFull n_states' triples emissions permutations' = do
  let n_events' = Vector.length emissions
  -- post is a contiguous mutable vector, convert it to an immutable 2D vector
  postFrozen <- shmm n_states' triples emissions permutations' False
  let postEmissions = flip Vector.map [0 .. (n_states'-1)] $ \ix ->
        Vector.convert $ Storable.slice (n_events' * ix) n_events' postFrozen

  --not ideal, not sure if it's avoidable - Eigen uses col-major
  let postEmissions' = flip Vector.map [0..(n_events'-1)] $ \ix -> Vector.map (Vector.! ix) postEmissions

  return postEmissions'


shmm ::
  -- The number of states, not including tokens
  Int ->
  -- Triples representing the transition distribution.
  -- Index 0 is reserved for the start token, other states start from 1.
  -- Column can go up to n_states+1, which represents the end token.
  [(Int, Int, Double)] ->
  -- Vector of observation distributions, one for each time point.
  -- Each distribution is n_obs long, which can be less than n_states.
  VecMat ->
  -- Vector to encode the permutation of states -> obervations.
  -- n_states long with values 0 through n_obs-1.
  Vector Int ->
  -- Whether or not to sum up the posteriors event-wise, returning a state vector
  Bool ->
  -- Returns a vector of posterior distributions over states, one for each event.
  -- n_events vectors of n_states each.
  IO (Storable.Vector Double)
shmm n_states' triples emissions permutations' summed' = do
  let n_triples = fromIntegral (length triples)
      n_states = fromIntegral n_states'
      n_obs = fromIntegral (Vector.length (Vector.head emissions))
      n_events' = Vector.length emissions
      n_events = fromIntegral (Vector.length emissions)
      permutations = (`Vector.snoc` (fromIntegral n_obs)) $ Vector.map fromIntegral permutations'
      summed = fromIntegral $ if summed' then 1 else 0

      --not ideal - where is best to add the 0s?
      emissions' = Vector.map (flip Vector.snoc 0) emissions

  -- initialize memory for the posterior, so we know where to read
  post <- if summed'
    then Mutable.unsafeNew (n_states' + 1)
    else Mutable.unsafeNew (n_events' * (n_states' + 1))

  -- call the C++ function, with temporary Ptrs to avoid a space leak
  withTripleArray triples $ \tsPtr ->
    withDoubleVector (Vector.toList emissions') $ \esPtr ->
      Storable.unsafeWith (Storable.convert permutations) $ \permPtr ->
        Mutable.unsafeWith post $ \postPtr ->
          c_shmm n_triples tsPtr n_states n_obs esPtr n_events permPtr summed postPtr

  -- post is a contiguous mutable vector, convert it to an immutable 2D vector
  postFrozen <- Storable.freeze post
  return postFrozen

-- C++ signature:
-- int shmm(int n_triples, DTriple *triples, int n_states, int n_obs, double **emissions_aptr, int n_events, int *permutation_, double *posterior_arr )
foreign import ccall "_Z4shmmiP7DTripleiiPPdiPibS1_"
  c_shmm :: CInt -> Ptr DTriple -> CInt -> CInt -> Ptr (Ptr Double) -> CInt -> Ptr CInt -> CUChar -> Ptr Double -> IO CInt

data DTriple = DTriple CInt CInt CDouble
type VecMat = Vector (Vector Double)

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

withVecMat :: VecMat -> (Ptr (Ptr Double) -> IO b) -> IO b
withVecMat emissions action = do
  ptrVec <- Vector.mapM (flip Storable.unsafeWith return . Storable.convert) emissions
  let storablePtrVec = Storable.convert ptrVec
  Storable.unsafeWith storablePtrVec action

withDoubleVector :: (Storable a) => [Vector a] -> (Ptr (Ptr a) -> IO b) -> IO b
withDoubleVector vs action = withDoubleVector' action vs []

withDoubleVector' :: (Storable a) => (Ptr (Ptr a) -> IO b) -> [Vector a] -> [Ptr a] -> IO b
withDoubleVector' action [] sofar = Storable.unsafeWith (Storable.fromList . reverse $ sofar) action
withDoubleVector' action (row:rest) sofar =
  Storable.unsafeWith (Storable.convert row) $ \rowPtr ->
                                                 withDoubleVector' action rest (rowPtr : sofar)
