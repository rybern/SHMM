module SHMM where

import Data.Map (Map)
import qualified Data.Map as Map
import EmissionIO
import Sequence
import System.Process
import System.IO.Temp
import Data.Monoid
import EmissionPermutation
import Data.Time

runHMM :: Map String Int
       -> FilePath
       -> MatSeq String
       -> IO Emissions
runHMM indexMap emissionsFile priorSeq = do
  withSystemTempDirectory "" $ \dir -> do
    let transFile = "transitions.st"
        permutationFile = dir<>"/permutation.csv"
        posteriorFile = dir<>"/posterior.csv"

    getZonedTime >>= print
    writeSTFile priorSeq transFile
    getZonedTime >>= print
    writeEmissionPerm indexMap permutationFile priorSeq
    getZonedTime >>= print
    callSHMM transFile emissionsFile permutationFile posteriorFile
    getZonedTime >>= print
    (Right posterior) <- readEmissions posteriorFile
    getZonedTime >>= print
    return posterior

callSHMM :: FilePath -> FilePath -> FilePath -> FilePath -> IO ()
callSHMM trans emms perm post = do
  callProcess "./shmm" [ trans, emms, post, perm ]
  return ()
