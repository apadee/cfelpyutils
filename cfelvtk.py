#    This file is part of cfelpyutils.
#
#    cfelpyutils is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    cfelpyutils is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with cfelpyutils.  If not, see <http://www.gnu.org/licenses/>.


"""
Utilties for the VTK python module.

Utilities for 3d data visualization using the Visualization Toolkit (VTK) python module.
"""


import numpy
import vtk

VTK_VERSION = vtk.vtkVersion().GetVTKMajorVersion()


def get_lookup_table(minimum_value, maximum_value, log=False, colorscale="jet", number_of_colors=1000):
    """Creates a vtkLookupTable object.

    Creates a vtkLookupTable object with a specified range and colorscale.

    Args:

        minimum_value (float): lowest value that the lookup table can display, lower values will be displayed as this
        value.

        maximum_value (float): highest value the the lookup table can display, higher values will be displayed as this
        value.

        log (Optional[bool]): if True, the scale used for the lookup table will be logarithmic.

        colorscale (Optional[string]): the name of any matplotlib colorscale. The lookuptable will replicate the
        specified colorscale.

        number_of_colors (Optional[int]): the length of the table. Longer tables result in smoother color scales.

    Returns:

        lookup_table (vtk.vtkLookupTable): A vtk lookup table object.
    """

    import matplotlib.cm
    if log:
        lut = vtk.vtkLogLookupTable()
    else:
        lut = vtk.vtkLookupTable()
    lut.SetTableRange(minimum_value, maximum_value)
    lut.SetNumberOfColors(number_of_colors)
    lut.Build()
    for i in range(number_of_colors):
        color = matplotlib.cm.cmap_d[colorscale](float(i) / float(number_of_colors))
        lut.SetTableValue(i, color[0], color[1], color[2], 1.)
    lut.SetUseBelowRangeColor(True)
    lut.SetUseAboveRangeColor(True)
    return lut


def array_to_float_array(array_in, dtype=None):
    """Converts a numpy array into a vtkFloatArray or vtkDoubleArray.

    Takes a numpy array and returns a vtkFloatArray or vtkDoubleArray, depending on the type of the input array.
    The array is flattened and thus the shape is lost.

    Args:

        array_in (numpy.ndarray): the array to convert.

        dtype (Optional[type]): if this argument is different from None, the array is converted to the specified data
        type. Otherwise the original type is preserved.

    Returns:

        float_array (vtk.vtkFloatArray): A float array of the specified type.
    """

    if dtype is None:
        dtype = array_in.dtype
    if dtype == "float32":
        float_array = vtk.vtkFloatArray()
    elif dtype == "float64":
        float_array = vtk.vtkDoubleArray()
    else:
        raise ValueError("Wrong format of input array, must be float32 or float64")
    if len(array_in.shape) == 2:
        float_array.SetNumberOfComponents(array_in.shape[1])
    elif len(array_in.shape) == 1:
        float_array.SetNumberOfComponents(1)
    else:
        raise ValueError("Wrong shape of array must be 1D or 2D.")
    float_array.SetVoidArray(numpy.ascontiguousarray(array_in, dtype), numpy.product(array_in.shape), 1)
    return float_array


def array_to_vtk(array_in, dtype=None):
    """Converts a numpy array into a vtk array of the specified type.

    Takes a numpy array and returns a vtk array of the specified type. The array is flattened and thus the shape is
    lost.

    Args:

        array_in (numpy.ndarray): the array to convert.

        dtype (Optional[type]): if this argument is different from None, the array is converted to the specified data
        type. Otherwise the original type is preserved.

    Returns:

        vtk_array (vtk.vtkFloatArray): a vtk float array of the specified type.
    """

    if dtype is None:
        dtype = numpy.dtype(array_in.dtype)
    else:
        dtype = numpy.dtype(dtype)
    if dtype == numpy.float32:
        vtk_array = vtk.vtkFloatArray()
    elif dtype == numpy.float64:
        vtk_array = vtk.vtkDoubleArray()
    elif dtype == numpy.uint8:
        vtk_array = vtk.vtkUnsignedCharArray()
    elif dtype == numpy.int8:
        vtk_array = vtk.vtkCharArray()
    else:
        raise ValueError("Wrong format of input array, must be float32 or float64")
    if len(array_in.shape) != 1 and len(array_in.shape) != 2:
        raise ValueError("Wrong shape: array must be 1D")
    vtk_array.SetNumberOfComponents(1)
    vtk_array.SetVoidArray(numpy.ascontiguousarray(array_in.flatten(), dtype), numpy.product(array_in.shape), 1)
    return vtk_array


def array_to_image_data(array_in, dtype=None):
    """Converts a numpy array to vtkImageData.

    Takes a numpy array and converts it to a vtkImageData object. The input must be a 3d array.

    Args:

        array_in (numpy.ndarray): array to be converted. Must be a 3d array.

        dtype (Optional[type]): if this argument is different from None, the array is converted to the specified data
        type. Otherwise the original type is preserved.

    Returns:

        image_data (vtk.vtkImageData): a vtk image data object containing data from the input array.
    """

    if len(array_in.shape) != 3:
        raise ValueError("Array must be 3D for conversion to vtkImageData")
    array_flat = array_in.flatten()
    float_array = array_to_float_array(array_flat, dtype)
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(*array_in.shape)
    image_data.GetPointData().SetScalars(float_array)
    return image_data


def window_to_png(render_window, file_name, magnification=1):
    """Saves a screen shot of a specific vt render window to a file.

    Saves a screenshot of a specific vtk render windows to a PNG format file.

    Args:

        render_window (vtk.vtkRenderWindow): the vt render window to capture.

        file_name (string): filename for the output file.

        magnification (Optional[int]): factor by which the resolution of the output file is multiplied.

    """
    magnification = int(magnification)
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.SetMagnification(magnification)
    window_to_image_filter.SetInputBufferTypeToRGBA()
    window_to_image_filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(file_name)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()


def poly_data_to_actor(poly_data, lut):
    """Creates a vtkActor object from a vtkPolyData object.

    Takes a vtkPolyData object and creates a vtkActor object from it. This circumvents the need to create a vtkMapper
    object by internally using a very basic vtkMapper.

    Args:

        poly_data (vtk.vtkPolyData): a vtkPolyData object.

        lut (vtk.vtkLookupTable): the colorscale to be used to map the input data into the vtk actor object.

    Returns:

        actor (vtk.vtkActor): a vtk actor object that can be used to display the input data
    """

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    mapper.SetLookupTable(lut)
    mapper.SetUseLookupTableScalarRange(True)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


class IsoSurface(object):
    """Isosurface objects.

    Class than implements the creation and display of isosurfaces, featuring one or more threshold levels.

    Args:

        volume (numpy.ndarray): a numpy 3d array.

        level (float | list of float): threshold level (or list of threshold levels) to be displayed by the isosurface
        object.
    """

    def __init__(self, volume, level=None):
        self._surface_algorithm = None
        self._renderer = None
        self._actor = None
        self._mapper = None
        self._volume_array = None

        self._float_array = vtk.vtkFloatArray()
        self._image_data = vtk.vtkImageData()
        self._image_data.GetPointData().SetScalars(self._float_array)
        self._setup_data(volume)

        self._surface_algorithm = vtk.vtkMarchingCubes()
        self._surface_algorithm.SetInputData(self._image_data)
        self._surface_algorithm.ComputeNormalsOn()

        if level is not None:
            try:
                self.set_multiple_levels(level)
            except TypeError:
                self.set_level(0, level)

        self._mapper = vtk.vtkPolyDataMapper()
        self._mapper.SetInputConnection(self._surface_algorithm.GetOutputPort())
        self._mapper.ScalarVisibilityOn()
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)

    def _setup_data(self, volume):
        """Creates self._volume_array and vtk array self._float_array objects.

         Creates self._volume_array and vtk array self._float_array objects from a numpy array, and makes sure that
         they share the same data.

        Args:

            volume (numpy.ndarray): numpy array used to populate the created objects.
        """

        self._volume_array = numpy.zeros(volume.shape, dtype="float32", order="C")
        self._volume_array[:] = volume
        self._float_array.SetNumberOfValues(numpy.product(volume.shape))
        self._float_array.SetNumberOfComponents(1)
        self._float_array.SetVoidArray(self._volume_array, numpy.product(volume.shape), 1)
        self._image_data.SetDimensions(*self._volume_array.shape)

    def set_renderer(self, renderer):
        """Sets the renderer for the isosurface object.

        Sets the renderer for the isosurface object (replacing any existing renderer).

        Args:

            renderer (vtk.vtkRenderer): renderer that will take control over all the vtk surface actors.
        """

        if self._actor is None:
            raise RuntimeError("Actor does not exist.")
        if self._renderer is not None:
            self._renderer.RemoveActor(self._actor)
        self._renderer = renderer
        self._renderer.AddActor(self._actor)

    def set_multiple_levels(self, levels):
        """Sets thresholds for the isosurface object.

        Sets thresholds for the isosurface object (replacing any existing thresholds).

        Args:

            levels (list of float): thresholds for the isosurface object. The thresholds must be in absolute values
            (not, for example, ratios).
        """

        self._surface_algorithm.SetNumberOfContours(0)
        for index, this_level in enumerate(levels):
            self._surface_algorithm.SetValue(index, this_level)
        self._render()

    def get_levels(self):
        """Returns current thresholds from an isosurface object.

        Returns a list of the thresholds used by an isosurface object.

        Returns:

            levels (list of floats): the thresholds currently used by the isosurface object.
        """

        return [self._surface_algorithm.GetValue(index)
                for index in range(self._surface_algorithm.GetNumberOfContours())]

    def add_level(self, level):
        """Adds a threshold to an isosurface object.

        Appends a single threshold to the list of thresholds used by an isosurface object.

        Args:
            level (float): the threshold that must be appended to the list of thresholds used by the isosurface
            obect.
        """

        self._surface_algorithm.SetValue(self._surface_algorithm.GetNumberOfContours(), level)
        self._render()

    def remove_level(self, index):
        """Removes threshold from an isosurface object.

        Removes a threshold from the list of thresholds used by an isosurface object. Requires the user to provide the
        index in the list of the threshold that must be removed.

        Args:

            index (int): the index in the threshold list of the threshold to remove. The indexes of the thresholds in
            the threshold lists usually match the order in which the thresholds were added to the list.
        """

        for index in range(index, self._surface_algorithm.GetNumberOfContours()-1):
            self._surface_algorithm.SetValue(index, self._surface_algorithm.GetValue(index+1))
        self._surface_algorithm.SetNumberOfContours(self._surface_algorithm.GetNumberOfContours()-1)
        self._render()

    def set_level(self, index, level):
        """Modifies a threshold in an isosurface object.

        Changes the value of a threshold in the threshold list used by an isosurface object.  Requires the user to
        provide the index in the list of the threshold that must be modified.

        Args:

            index (int): the index in the threshold list of the threshold to modify. The indexes of the thresholds in
            the threshold lists usually match the order in which the thresholds were added to the list.

            level (float): the new value for the threshold.
        """

        self._surface_algorithm.SetValue(index, level)
        self._render()

    def set_cmap(self, cmap):
        """Sets colormap for an isosurface object.

        Set the colormap that an isosurface object uses to plot isosurfaces. When a colomap is set and the object plots
        multiple isosurfaces, the color of each isosurface is chosen by comparing the threshold level to the color map.
        This is relevant mainly when the object plots multiple isosurfaces.

        Args:

            cmap (string): name of the colormap used to plot the isosurfaces. Can be any colormap provided by
            matplotlib.
        """

        self._mapper.ScalarVisibilityOn()
        self._mapper.SetLookupTable(get_lookup_table(self._volume_array.min(), self._volume_array.max(),
                                                     colorscale=cmap))
        self._render()

    def set_color(self, color):
        """Sets color for an isosurface object.

        Sets a color that the isosurface object will use to plot all isosurfaces.

        Args:

            color (tuple of int): a size 3 tuple with the RGB value of the color.
        """

        self._mapper.ScalarVisibilityOff()
        self._actor.GetProperty().SetColor(color[0], color[1], color[2])
        self._render()

    def set_opacity(self, opacity):
        """Sets threshold opacity for an isosurface object.

        Sets the opacity used by an isosurface object to plot all isosurfaces. (Setting the opacity of each
        individual isosurface is not supported).

        Args:

            opacity (float): a number between between 0. and 1, where 0. means that the surface is completely
            transparent and 1. means that the surface is completely opaque.
        """

        self._actor.GetProperty().SetOpacity(opacity)
        self._render()

    def _render(self):
        """Render isosurfaces.

        Renders all isosurfaces in an isosurface object if a renderer is set, otherwise does nothing.
        """

        if self._renderer is not None:
            self._renderer.GetRenderWindow().Render()

    def set_data(self, volume):
        """Sets the dataset that the isosurface object displays.

        Sets the dataset that the isosurface objects displays. Replaces any existing data set.

        Args:

            volume (numpy.ndarray): the array containing the new data set to be displayed. It must have the same shape
            as the old data array.
        """

        if volume.shape != self._volume_array.shape:
            raise ValueError("New volume must be the same shape as the old one")
        self._volume_array[:] = volume
        self._float_array.Modified()
        self._render()


def plot_isosurface(volume, level=None, opacity=1.):
    """Plots one or more isosurfaces for the provided volume dataset.

    Plots one of more isosurfaces for the provided volume dataset.


    Args:

        volume (numpy.ndarray): the 3d array containg the dataset to be displayed.

        level (float | list of float): isosurface threshold to display (or list of isosurface thresholds, when
        displaying multiple thresholds).

        opacity (float): opacity used to plot the isosurfaces. A number between 0. and 1, where 0. means that the
        isosurfaces will be completely transparent and 1. means that the isosurfaces will be completely opaque.
    """

    surface_object = IsoSurface(volume, level)
    surface_object.set_opacity(opacity)

    renderer = vtk.vtkRenderer()
    if opacity != 1.:
        renderer.SetUseDepthPeeling(True)
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandPick())

    surface_object.set_renderer(renderer)

    renderer.SetBackground(0., 0., 0.)
    render_window.SetSize(800, 800)
    interactor.Initialize()
    render_window.Render()
    interactor.Start()


def plot_planes(array_in, log=False, cmap=None):
    """Plots the volume at the intersection of two planes.

    Displays the volume at the intersection of two planes that cut through a volume. The planes can be manipulated
    interactively.

    Args:

        array_in (numpy.ndarray): the 3d array containg the volume data to be displayed.

        log (bool): if True, the data will be displayed in logarithmic scale.

        cmap (string): the colormap to be sued use for displaying the data. Can be any colormap provided by matplotlib.
    """

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandPick())

    if cmap is None:
        import matplotlib as _matplotlib
        cmap = _matplotlib.rcParams["image.cmap"]
    lut = get_lookup_table(max(0., array_in.min()), array_in.max(), log=log, colorscale=cmap)
    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.005)

    image_data = array_to_image_data(array_in.astype(numpy.float64))

    def setup_plane():
        """Create and setup a singel plane."""
        plane = vtk.vtkImagePlaneWidget()
        if VTK_VERSION >= 6:
            plane.SetInputData(image_data)
        else:
            plane.SetInput(image_data)
        plane.UserControlledLookupTableOn()
        plane.SetLookupTable(lut)
        plane.DisplayTextOn()
        plane.SetPicker(picker)
        plane.SetLeftButtonAction(1)
        plane.SetMiddleButtonAction(2)
        plane.SetRightButtonAction(0)
        plane.SetInteractor(interactor)
        return plane

    plane_1 = setup_plane()
    plane_1.SetPlaneOrientationToXAxes()
    plane_1.SetSliceIndex(array_in.shape[0]//2)
    plane_1.SetEnabled(1)
    plane_2 = setup_plane()
    plane_2.SetPlaneOrientationToYAxes()
    plane_2.SetSliceIndex(array_in.shape[1]//2)
    plane_2.SetEnabled(1)

    renderer.SetBackground(0., 0., 0.)
    render_window.SetSize(800, 800)
    interactor.Initialize()
    render_window.Render()
    interactor.Start()


def setup_window(size=(400, 400), background=(1., 1., 1.)):
    """Sets up a rendering window.

    Creates renderer, render_window and interactor objects, and sets up the connections between them.

    Args:

        size (Optional[tuple of int]): size 2 tuple describing the size of the window in pixels. If the user does
        not provide this argument, the created window will have a size of 400x400 pixels.

        background (Optional[tuple of float]): a size 3 tuple with the RGB value of the window background color. If
        the user does not provide this argument, the background will be white.

    Returns:

        renderer (vtk.vtkRenderer): a standard renderer object connected to the the a vtk window object.

        render_window (vtk.vtkRenderWindow): a vtk render window object with the dimensions requested by the user.

        interactor (vtk.vtkRenderWindowInteractor): an vtk interactor object using the rubber band pick interactor
        style.
    """

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleRubberBandPick())
    interactor.SetRenderWindow(render_window)

    renderer.SetBackground(background[0], background[1], background[2])
    render_window.SetSize(size[0], size[1])

    interactor.Initialize()
    render_window.Render()
    return renderer, render_window, interactor


def scatterplot_3d(data, color=None, point_size=None, point_shape=None):
    """Displays a 3d scatter plot.

    Displays a scatter plot in 3 dimensions.

    Args:

        data (numpy.ndimage): the array with the data to be plotted. The array must have a shape of Nx3, where N is the
        number of points in the data set, and the other dimensions are the coordinates of earch data point.

        color (Optional[numpy.ndimage]): 1d array of floating points of with same length as the data array (N),
        describing the color used to plot each data point.

        point_size (Optional[float]): The size factor for the points in the scatterplot. Behaves differently depending
        on the value of the point_shape argument (See below). If "spheres" are used for point_shape argument, the size
        factor is relative to the size of the scene, if "squares" is used, the size factor is relative to the size of
        the window. If the user does not specify any size factor, a reasonable value will be picked automatically.

        point_shape (Optional[str]): can be "spheres" or "squares". The former plots each data point as a 3d sphere
        (recommended only for small data sets), while the latter displays each point as a 2d square without any 3d
        structure. If the user does not specify any point_shape, "spheres" will be automatically picked for data sets
        with less than 1000 elements, while "squares" will be automatically picked for larger datasets.
    """

    if len(data.shape) != 2 or data.shape[1] != 3:
        raise ValueError("data must have shape (n, 3) where n is the number of points.")
    if point_shape is None:
        if len(data) <= 1000:
            point_shape = "spheres"
        else:
            point_shape = "squares"
    data_vtk = array_to_float_array(data.astype(numpy.float32))
    point_data = vtk.vtkPoints()
    point_data.SetData(data_vtk)
    points_poly_data = vtk.vtkPolyData()
    points_poly_data.SetPoints(point_data)

    if point_shape == "spheres":
        if point_size is None:
            point_size = numpy.array(data).std() / len(data)**(1./3.) / 3.
        glyph_filter = vtk.vtkGlyph3D()
        glyph_filter.SetInputData(points_poly_data)
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(point_size)
        glyph_filter.SetSourceConnection(sphere_source.GetOutputPort())
        glyph_filter.SetScaleModeToDataScalingOff()
        if color is not None:
            glyph_filter.SetColorModeToColorByScalar()
        else:
            glyph_filter.SetColorMode(0)
        glyph_filter.Update()
    elif point_shape == "squares":
        if point_size is None:
            point_size = 3
        glyph_filter = vtk.vtkVertexGlyphFilter()
        glyph_filter.SetInputData(points_poly_data)
        glyph_filter.Update()
    else:
        raise ValueError("{0} is not a valid entry for points".format(point_shape))

    poly_data = vtk.vtkPolyData()
    poly_data.ShallowCopy(glyph_filter.GetOutput())

    renderer, render_window, interactor = setup_window()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    if color is not None:
        lut = get_lookup_table(color.min(), color.max())
        color_scalars = array_to_vtk(color.copy().astype(numpy.float32))
        color_scalars.SetLookupTable(lut)
        points_poly_data.GetPointData().SetScalars(color_scalars)
        mapper.SetLookupTable(lut)
        mapper.SetUseLookupTableScalarRange(True)

    points_actor = vtk.vtkActor()
    points_actor.SetMapper(mapper)
    points_actor.GetProperty().SetPointSize(point_size)
    points_actor.GetProperty().SetColor(0., 0., 0.)

    axes_actor = vtk.vtkCubeAxesActor()
    axes_actor.SetBounds(points_actor.GetBounds())
    axes_actor.SetCamera(renderer.GetActiveCamera())
    axes_actor.SetFlyModeToStaticTriad()
    axes_actor.GetXAxesLinesProperty().SetColor(0., 0., 0.)
    axes_actor.GetYAxesLinesProperty().SetColor(0., 0., 0.)
    axes_actor.GetZAxesLinesProperty().SetColor(0., 0., 0.)
    for i in range(3):
        axes_actor.GetLabelTextProperty(i).SetColor(0., 0., 0.)
        axes_actor.GetTitleTextProperty(i).SetColor(0., 0., 0.)

    renderer.AddActor(points_actor)
    renderer.AddActor(axes_actor)

    render_window.Render()
    interactor.Start()
