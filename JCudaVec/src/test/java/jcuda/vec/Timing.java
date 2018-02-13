/*
 * JCudaVec - Vector operations for JCuda 
 * http://www.jcuda.org
 *
 * Copyright (c) 2013-2018 Marco Hutter - http://www.jcuda.org
 * 
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
package jcuda.vec;

import java.util.Locale;

/**
 * Utility class storing timing information for the benchmarks
 */
@SuppressWarnings("javadoc")
class Timing
{
    private String name;
    private long beforeHost;
    private long afterHost;
    private long beforeDeviceTotal;
    private long beforeDeviceCore;
    private long afterDeviceCore;
    private long afterDeviceTotal;
    private double hostDurationMs; 
    private double deviceTotalDurationMs; 
    private double deviceCoreCurationMs;
    
    Timing(String name)
    {
        this.name = name;
    }
    
    String getName()
    {
        return name;
    }
    
    void startHost()
    {
        beforeHost = System.nanoTime();
    }
    void endHost()
    {
        afterHost = System.nanoTime();
        hostDurationMs = (afterHost - beforeHost) / 1e6; 
    }
    void startDeviceTotal()
    {
        beforeDeviceTotal = System.nanoTime();
    }
    void startDeviceCore()
    {
        beforeDeviceCore = System.nanoTime();
    }
    void endDeviceCore()
    {
        afterDeviceCore = System.nanoTime();
        deviceCoreCurationMs = (afterDeviceCore - beforeDeviceCore) / 1e6;
    }
    void endDeviceTotal()
    {
        afterDeviceTotal = System.nanoTime();
        deviceTotalDurationMs = (afterDeviceTotal - beforeDeviceTotal) / 1e6;
    }
    
    
    @Override
    public String toString()
    {
        return createString(name.length());
    }
    
    String createString(int nameWidth)
    {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-" + nameWidth + "s ", name));
        sb.append(format(hostDurationMs));
        sb.append(format(deviceCoreCurationMs));
        sb.append(format(deviceTotalDurationMs));
        return sb.toString();
    }
    
    private static String format(double ms)
    {
        String numberFormat = "%7.2f";
        String numberString = 
            String.format(Locale.ENGLISH, numberFormat, ms);
        return String.format("%10s", numberString);
    }
    
    /**
     * Print a formatted string representation of the given timings
     * to the standard output
     * 
     * @param timings The Timings
     */
    static void print(Iterable<? extends Timing> timings)
    {
        System.out.println(createString(timings));
    }
    
    /**
     * Create a formatted string representation of the given timings
     * 
     * @param timings The timings
     * @return The string
     */
    private static String createString(Iterable<? extends Timing> timings)
    {
        int maxNameLength = -1;
        for (Timing timing : timings)
        {
            maxNameLength = Math.max(maxNameLength, timing.getName().length());
        }
        int columnWidth = 10;
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-" + maxNameLength + "s ", "Name"));
        sb.append(String.format("%" + columnWidth + "s", "Host"));
        sb.append(String.format("%" + columnWidth + "s", "Dev."));
        sb.append(String.format("%" + columnWidth + "s", "Dev.Tot."));
        sb.append("\n");
        for (Timing timing : timings)
        {
            sb.append(timing.createString(maxNameLength));
            sb.append("\n");
        }
        return sb.toString();
    }
    
}