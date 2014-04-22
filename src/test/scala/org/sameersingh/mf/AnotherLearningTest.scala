package org.sameersingh.mf

import scala.util.Random

import org.junit._
import org.sameersingh.mf.learner.BatchTrainer
import org.sameersingh.mf.learner.SGDTrainer
import org.sameersingh.mf.learner.Trainer

class AnotherLearningTest {
	implicit val random = new Random(1)

  def smallSingleMatrix(numComps: Int, noiseVar: Double = 0.0) = {
    val numRows = 4
    val numCols = 4
    val trainingProp = 1.0
    val rowFactors = new DoubleDenseMatrix("r_t", numComps)
    val colFactors = new DoubleDenseMatrix("c_t", numComps)
    for (k <- 0 until numComps) {
      for (i <- 0 until numRows) {
        rowFactors(SimpleID(i, "r"), k) = 0 // random.nextGaussian() / math.sqrt(numComps.toDouble)
      }
      for (j <- 0 until numCols) {
        colFactors(SimpleID(j, "c"), k) = 0 //random.nextGaussian() / math.sqrt(numComps.toDouble)
      }
    }

    val matrix = new Matrix("testMatrix")
    for (i <- 0 until numRows)
      for (j <- 0 until numCols) {
        val rid = SimpleID(i, "r")
        val cid = SimpleID(j, "c")
//        val doubleValue = rowFactors.r(rid).zip(colFactors.r(cid)).map(uv => uv._1 * uv._2).sum + noiseVar * random.nextGaussian()
        val doubleValue =  if (i+j < 2 || i+j > 4 || i==j) 1 else 0
        val cell = new Cell {
          val row = rid

          val col = cid

          val value = DoubleValue(doubleValue)

          val isTrain = random.nextDouble() < trainingProp

          val inMatrix = matrix
        }
        matrix += cell
      }
    matrix
  }

  def genParams(numComps: Int) = {
    val params = new ParameterSet
    params += new DoubleDenseMatrix("r", numComps, () => random.nextGaussian() / 100.0)
    params += new DoubleDenseMatrix("c", numComps, () => random.nextGaussian() / 100.0)
    params(params[DoubleDenseMatrix]("r"), "bias") = (() => random.nextGaussian() / 100.0)
    params(params[DoubleDenseMatrix]("c"), "bias") = (() => random.nextGaussian() / 100.0)
    params(params[DoubleDenseMatrix]("r"), "L2RegCoeff") = 100
    params(params[DoubleDenseMatrix]("c"), "L2RegCoeff") = 100
    params
  }

  def terms(m: ObservedMatrix) = {
    val params = genParams(3)
    val term = new DotL2(params, "r", "c", 1.0, m)
    val l2r = new L2Regularization(params, "r", m.trainCells.size)
    val l2c = new L2Regularization(params, "c", m.trainCells.size)
    (term, l2r, l2c)
  }

  def train(trainer: Trainer, term: DotL2, l2r: L2Regularization, l2c: L2Regularization, m: ObservedMatrix) {
    for (i <- 0 until 500) {
      trainer.round(0.01)
    }
    println(term.evalTrain(m).mkString("\n"))
    println(term.evalTest(m).mkString("\n"))
  }

  @Test
  def sgdTrainingSmallSingle() {
//    for (i <- 0 until 1) {
      println(" --- SGD")// %d ---" format(i+1))
      val m = smallSingleMatrix(3, 0.01)
      //println(m)
      val (term, l2r, l2c) = terms(m)
      val trainer = new SGDTrainer(Seq(m), Seq(term, l2r, l2c))
      train(trainer, term, l2r, l2c, m)
      val v1 = term.params(0).asInstanceOf[DoubleDenseMatrix]
      val v2 = term.params(1).asInstanceOf[DoubleDenseMatrix]
        DoubleDenseMatrix.save(v1, "v"+v1.name, false)
        DoubleDenseMatrix.save(v2, "v"+v2.name, false)
      
//      assertEquals(0.0, term.avgValue(m.trainCells), 0.001)
//      assertEquals(0.0, term.avgValue(m.testCells), 0.0025)
//    }
  }
}