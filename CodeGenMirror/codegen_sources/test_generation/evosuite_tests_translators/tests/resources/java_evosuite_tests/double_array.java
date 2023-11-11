/*
 * This file was automatically generated by EvoSuite
 * Tue Jan 18 23:43:16 GMT 2022
 */


import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true)
public class CLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba_ESTest extends CLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      double[] doubleArray0 = new double[7];
      doubleArray0[0] = (-1.0);
      double double0 = CLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba.avgOfArray(doubleArray0);
      assertEquals((-0.14285714285714285), double0, 1.0E-4);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      double double0 = CLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba.avgOfArray(doubleArray0);
      assertEquals(0.0, double0, 1.0E-4);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      double double0 = CLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba.avgOfArray(doubleArray0);
      assertEquals(0.0, double0, 1.0E-4);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = (-56615.2289);
      double double0 = CLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba.avgOfArray(doubleArray0);
      assertEquals((-56615.2289), double0, 1.0E-4);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      doubleArray0[0] = 19833.7892;
      double double0 = CLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba.avgOfArray(doubleArray0);
      assertEquals(19833.7892, double0, 1.0E-4);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      CLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba cLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba0 = new CLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba();
  }

    @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Double[] doubleArray0 = new Double[0];
      Double double0 = CLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba.avgOfArray(doubleArray0);
      assertEquals(0.0, (double)double0, 1.0E-4);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Double[] doubleArray0 = new Double[4];
      Double double0 = new Double((-1.0));
      doubleArray0[0] = double0;
      Double double1 = new Double(91376.784102651);
      doubleArray0[1] = double1;
      doubleArray0[2] = double0;
      doubleArray0[3] = double0;
      Double double2 = CLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba.avgOfArray(doubleArray0);
      assertEquals(91373.784102651, (double)double2, 1.0E-4);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Double[] doubleArray0 = new Double[4];
      Double double0 = new Double((-1.0));
      doubleArray0[0] = double0;
      Double[] doubleArray1 = new Double[2];
      doubleArray1[0] = double0;
      doubleArray1[1] = doubleArray0[0];
      Double double1 = CLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba.avgOfArray(doubleArray1);
      assertEquals((-2.0), (double)double1, 1.0E-4);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      Double[] doubleArray0 = new Double[1];
      Double double0 = new Double(0.0);
      doubleArray0[0] = double0;
      Double double1 = CLASS_005ae0a2dee4fd5b48473a73a35ef9e0ba70f2b9958874819beb1bd7c37bd2ba.avgOfArray(doubleArray0);
      assertEquals(0.0, (double)double1, 1.0E-4);
  }


}
