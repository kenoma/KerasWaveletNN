import unittest
from wnn_layer import WNN
from keras import backend as K

class WNNLayerTests(unittest.TestCase):
    """
    Test values was obtained from manual calculations
    """

    def test_layer_output_single(self):
        odim = 1
        input_dim = 1
        batch = 1
        wavelons = 1
        sess = K.get_session()

        layer = WNN(wavelons, odim)
        layer.build(input_shape=(batch, input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        w = K.placeholder(shape=(odim, wavelons))
        bias = K.placeholder(shape=(odim,1))
        dilation = K.placeholder(shape=(wavelons, input_dim))
        translation = K.placeholder(shape=(wavelons, input_dim))

        layer.w = w
        layer.bias = bias
        layer.dilation = dilation
        layer.translation = translation

        xc = layer.call(x)

        xx = [[1]]
        ww = [[1]]
        bb = [[0]]
        dd = [[0.3]]
        tt = [[0.5]]

        vals = sess.run(xc, feed_dict={x: xx, w:ww, bias:bb, dilation:dd, translation:tt })
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], -0.4155870142, 7)

    def test_layer_output_single2(self):
        odim = 1
        input_dim = 1
        batch = 1
        wavelons = 1
        sess = K.get_session()

        layer = WNN(wavelons, odim)
        layer.build(input_shape=(batch, input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        w = K.placeholder(shape=(odim, wavelons))
        bias = K.placeholder(shape=(odim,1))
        dilation = K.placeholder(shape=(wavelons, input_dim))
        translation = K.placeholder(shape=(wavelons, input_dim))

        layer.w = w
        layer.bias = bias
        layer.dilation = dilation
        layer.translation = translation

        xc = layer.call(x)

        xx = [[1]]
        ww = [[1]]
        bb = [[1]]
        dd = [[0.3]]
        tt = [[0.5]]

        vals = sess.run(xc, feed_dict={x: xx, w:ww, bias:bb, dilation:dd, translation:tt })
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], -0.4155870142 + 1, 7)

    def test_layer_output_single3(self):
        odim = 1
        input_dim = 1
        batch = 1
        wavelons = 1
        sess = K.get_session()

        layer = WNN(wavelons, odim)
        layer.build(input_shape=(batch, input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))

        w = K.placeholder(shape=(odim,wavelons))
        bias = K.placeholder(shape=(odim,1))
        dilation = K.placeholder(shape=(wavelons, input_dim))
        translation = K.placeholder(shape=(wavelons, input_dim))

        layer.w = w
        layer.bias = bias
        layer.dilation = dilation
        layer.translation = translation

        xc = layer.call(x)

        xx = [[1]]
        ww = [[3.14]]
        bb = [[1]]
        dd = [[0.3]]
        tt = [[0.5]]

        vals = sess.run(xc, feed_dict={x: xx, w:ww, bias:bb, dilation:dd, translation:tt })
        self.assertEqual(len(vals), batch)
        self.assertAlmostEqual(vals[0][0], 3.14 * -0.4155870142 + 1, 7)
        self.assertAlmostEqual(vals[0][0], 3.14 * -0.4155870142 + 1, 7)

    def test_layer_two_dimensional_input(self):
        odim = 1
        input_dim = 2
        batch = 1
        wavelons = 1
        sess = K.get_session()

        layer = WNN(wavelons, odim)
        layer.build(input_shape=(batch, input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        w = K.placeholder(shape=(odim, wavelons))
        bias = K.placeholder(shape=(odim,1))
        dilation = K.placeholder(shape=(wavelons, input_dim))
        translation = K.placeholder(shape=(wavelons, input_dim))

        layer.w = w
        layer.bias = bias
        layer.dilation = dilation
        layer.translation = translation

        xc = layer.call(x)

        xx = [[1,2]]
        ww = [[3]]
        bb = [[4]]
        dd = [[5,6]]
        tt = [[7,8]]

        vals = sess.run(xc, feed_dict={x: xx, w:ww, bias:bb, dilation:dd, translation:tt })
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 5.819592, 7)

    def test_layer_two_dimensional_input_two_wavelons(self):
        odim = 1
        input_dim = 2
        batch = 1
        wavelons = 2
        sess = K.get_session()

        layer = WNN(wavelons, odim)
        layer.build(input_shape=(batch, input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        w = K.placeholder(shape=(odim, wavelons))
        bias = K.placeholder(shape=(odim, 1))
        dilation = K.placeholder(shape=(wavelons, input_dim))
        translation = K.placeholder(shape=(wavelons, input_dim))

        layer.w = w
        layer.bias = bias
        layer.dilation = dilation
        layer.translation = translation

        xc = layer.call(x)

        xx = [[1, 2]]
        ww = [[3, 4]]
        bb = [[4]]
        dd = [[5, 6],[6, 7]]
        tt = [[7, 8],[8, 9]]

        vals = sess.run(xc, feed_dict={x: xx, w:ww, bias:bb, dilation:dd, translation:tt })
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 8.245714, 5)

    def test_layer_two_dimensional_input_two_wavelons_two_dimensional_output(self):
        odim = 2
        input_dim = 2
        batch = 1
        wavelons = 2
        sess = K.get_session()

        layer = WNN(wavelons, odim)
        layer.build(input_shape=(batch, input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        w = K.placeholder(shape=(odim, wavelons))
        bias = K.placeholder(shape=(1, odim))
        dilation = K.placeholder(shape=(wavelons, input_dim))
        translation = K.placeholder(shape=(wavelons, input_dim))

        layer.w = w
        layer.bias = bias
        layer.dilation = dilation
        layer.translation = translation

        xc = layer.call(x)

        xx = [[1, 2]]
        ww = [[3, 4], [5, 6]]
        bb = [[4, 5]]
        dd = [[5, 6],[6, 7]]
        tt = [[7, 8],[8, 9]]

        vals = sess.run(xc, feed_dict={x: xx, w:ww, bias:bb, dilation:dd, translation:tt })
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 8.245714, 5)
        self.assertAlmostEqual(vals[0][1], 11.671837, 5)

    def test_layer_two_dimensional_input_two_wavelons_two_dimensional_output_three_batches(self):
        odim = 2
        input_dim = 2
        batch = 3
        wavelons = 2
        sess = K.get_session()

        layer = WNN(wavelons, odim)
        layer.build(input_shape=(batch, input_dim))
        
        x = K.placeholder(shape=(batch, input_dim))
        w = K.placeholder(shape=(odim, wavelons))
        bias = K.placeholder(shape=(1, odim))
        dilation = K.placeholder(shape=(wavelons, input_dim))
        translation = K.placeholder(shape=(wavelons, input_dim))

        layer.w = w
        layer.bias = bias
        layer.dilation = dilation
        layer.translation = translation

        xc = layer.call(x)

        xx = [[1, 2],[3,4],[5,6]]
        ww = [[3, 4], [5, 6]]
        bb = [[4, 5]]
        dd = [[5, 6],[6, 7]]
        tt = [[7, 8],[8, 9]]

        vals = sess.run(xc, feed_dict={x: xx, w:ww, bias:bb, dilation:dd, translation:tt })
        self.assertEqual(len(vals), batch)
        self.assertEqual(len(vals[0]), odim)
        self.assertAlmostEqual(vals[0][0], 8.245714, 5)
        self.assertAlmostEqual(vals[0][1], 11.671837, 5)
        self.assertEqual(len(vals[1]), odim)
        self.assertAlmostEqual(vals[1][0], 8.098251, 5)
        self.assertAlmostEqual(vals[1][1], 11.437838, 5)
        self.assertEqual(len(vals[2]), odim)
        self.assertAlmostEqual(vals[2][0], 6.8727336, 5)
        self.assertAlmostEqual(vals[2][1], 9.493723, 5)

if __name__ == '__main__':
    unittest.main()
